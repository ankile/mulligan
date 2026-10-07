"""Quota accounting for blinded multi-arm collection (sim and real collectors).

A manifest lists the start states of a round, each with the arms (``sources``)
it belongs to; one collected episode can credit several arms. The ledger is an
append-only JSONL file, one row per saved episode, so a session resumes from it
and the splitters (:mod:`mulligan.data.split_protocol_quota`) rebuild the
per-arm datasets from it.

:class:`ProtocolQuotaLedger` keeps simultaneous no-CF and with-CF quotas per
arm (the protocol of the paper rounds). Counterfactual replays count as
distinct episodes and credit with-CF only.
"""

from __future__ import annotations

import json
import hashlib
import time
from collections import Counter
from collections import deque
from pathlib import Path
from typing import Any

import numpy as np

from mulligan.utils.manifest_matching import ManifestMatcher, angular_idxs_for_keys


def parse_arm_target_caps(spec: str | None) -> dict[str, dict[str, int | str]] | None:
    """Parse ``PROTOCOL.ARM=INT|collected[;...]`` (the collector / splitter ``--protocol-quota-arm-caps``
    literal). ``collected`` freezes that arm's target at the count already in the ledger."""
    if spec is None or not spec.strip():
        return None
    caps: dict[str, dict[str, int | str]] = {}
    for item in spec.split(";"):
        item = item.strip()
        if not item:
            continue
        if "=" not in item or "." not in item.split("=", 1)[0]:
            raise ValueError(f"arm cap entries must be PROTOCOL.ARM=INT|collected; got {item!r}")
        key, raw = item.split("=", 1)
        protocol, arm = (part.strip() for part in key.split(".", 1))
        raw = raw.strip()
        if not protocol or not arm or not raw:
            raise ValueError(f"arm cap entry has an empty field: {item!r}")
        value: int | str = "collected" if raw == "collected" else int(raw)
        if arm in caps.get(protocol, {}):
            raise ValueError(f"duplicate arm cap for {protocol}.{arm}")
        caps.setdefault(protocol, {})[arm] = value
    return caps or None


class ProtocolQuotaLedger:
    """Track protocol x arm quotas for simultaneous no-CF and with-CF collection.

    Fresh successful episodes may credit both protocols. Counterfactual replays
    may credit only the with-CF protocol. Protocols may target different arm
    sets; for example, a baseline arm can be no-CF only while Ours arms fill
    both no-CF and with-CF quotas. A per-protocol balance slack prevents one
    targeted arm from running far ahead of the others, which keeps the with-CF
    quota from finishing one arm at a time.
    """

    def __init__(
        self,
        *,
        manifest_path: Path,
        targets_by_protocol: dict[str, int],
        ledger_path: Path,
        arms_by_protocol: dict[str, list[str]] | None = None,
        balance_slack: int = 1,
        progress_window: int = 20,
        selection_mode: str = "hard_balance",
        softmax_beta: float = 1.0,
        sampling_seed: int = 0,
        arm_target_caps: dict[str, dict[str, int | str]] | None = None,
    ) -> None:
        self.manifest_path = Path(manifest_path)
        self.ledger_path = Path(ledger_path)
        self.targets_by_protocol = {
            str(protocol): int(target) for protocol, target in targets_by_protocol.items()
        }
        if not self.targets_by_protocol:
            raise ValueError("targets_by_protocol must not be empty")
        required_protocols = {"no_cf", "with_cf"}
        missing_protocols = required_protocols - set(self.targets_by_protocol)
        if missing_protocols:
            raise ValueError(
                "ProtocolQuotaLedger requires no_cf and with_cf targets; "
                f"missing {sorted(missing_protocols)}"
            )
        for protocol, target in self.targets_by_protocol.items():
            if target <= 0:
                raise ValueError(f"target for {protocol!r} must be positive, got {target}")

        self.balance_slack = int(balance_slack)
        if self.balance_slack < 0:
            raise ValueError(f"balance_slack must be >= 0, got {balance_slack}")
        self.progress_window = int(progress_window)
        if self.progress_window <= 0:
            raise ValueError(f"progress_window must be positive, got {progress_window}")
        self.selection_mode = str(selection_mode)
        if self.selection_mode not in {"hard_balance", "soft_weighted"}:
            raise ValueError(
                f"selection_mode must be 'hard_balance' or 'soft_weighted', got {selection_mode!r}"
            )
        self.softmax_beta = float(softmax_beta)
        if not np.isfinite(self.softmax_beta) or self.softmax_beta < 0.0:
            raise ValueError(f"softmax_beta must be finite and >= 0, got {softmax_beta}")
        self.sampling_seed = int(sampling_seed)

        payload = json.loads(self.manifest_path.read_text())
        # Optional per-(protocol, arm) target CAPS declared by the manifest
        # (``protocol_arm_targets``: {protocol: {arm: target}}), used for a ledgered
        # mid-collection amendment such as "stop chasing the ours no-CF quota once with-CF
        # is served". A cap lowers that arm's target below the protocol target; it never
        # raises it. Fresh selection then draws from the UNION of protocols an arm still
        # owes (see ``fresh_union_selection``) so the arm interleave survives the cap.
        # The cap may also come from the collector CLI (``arm_target_caps``), where the value
        # ``"collected"`` means "freeze this arm's target at the count already in the ledger"
        # -- a pure accounting rule resolved at resume, so the manifest bytes stay untouched.
        raw_caps: dict[str, dict[str, int | str]] = {
            str(protocol): {str(arm): target for arm, target in caps.items()}
            for protocol, caps in (payload.get("protocol_arm_targets") or {}).items()
        }
        for protocol, caps in (arm_target_caps or {}).items():
            for arm, target in caps.items():
                manifest_value = raw_caps.get(str(protocol), {}).get(str(arm))
                if manifest_value is not None and manifest_value != target:
                    raise ValueError(
                        f"arm cap for {protocol}.{arm}: CLI {target!r} != manifest "
                        f"{manifest_value!r}"
                    )
                raw_caps.setdefault(str(protocol), {})[str(arm)] = target
        self.protocol_arm_targets: dict[str, dict[str, int | str]] = raw_caps
        for protocol, caps in self.protocol_arm_targets.items():
            if protocol not in self.targets_by_protocol:
                raise ValueError(
                    f"{self.manifest_path}: arm cap names unknown protocol {protocol!r}"
                )
            for arm, target in caps.items():
                if target == "collected":
                    continue
                if (
                    not isinstance(target, int)
                    or target < 0
                    or target > self.targets_by_protocol[protocol]
                ):
                    raise ValueError(
                        f"arm cap [{protocol!r}][{arm!r}]={target!r} must be 'collected' or an "
                        f"int in [0, {self.targets_by_protocol[protocol]}] (a cap may only lower "
                        "a protocol target)"
                    )
        self.fresh_union_selection = bool(self.protocol_arm_targets)
        self.task = str(payload["task"])
        self.keys = list(payload["keys"])
        self.match_tolerance = float(payload.get("match_tolerance", 1e-3))
        self.states = list(payload["states"])
        if not self.states:
            raise ValueError(f"{self.manifest_path}: manifest has no states")
        self.manifest_hash = hashlib.sha256(
            json.dumps(
                {
                    "task": self.task,
                    "keys": self.keys,
                    "match_tolerance": self.match_tolerance,
                    "states": self.states,
                },
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()

        self.manifest_arr = np.array(
            [[float(s[k]) for k in self.keys] for s in self.states],
            dtype=np.float64,
        )
        self._angular_idxs = angular_idxs_for_keys(self.keys)
        self.sources_per_state = [list(s["sources"]) for s in self.states]
        empty_source_idxs = [
            idx for idx, sources in enumerate(self.sources_per_state) if not sources
        ]
        if empty_source_idxs:
            raise ValueError(
                f"{self.manifest_path}: every state must list at least one source; "
                f"empty sources at indices {empty_source_idxs[:10]}"
            )
        self._matcher = ManifestMatcher(self.manifest_arr, self._angular_idxs)
        self.arms = sorted({src for srcs in self.sources_per_state for src in srcs})
        if arms_by_protocol is None:
            self.arms_by_protocol = {
                protocol: list(self.arms) for protocol in self.targets_by_protocol
            }
        else:
            unknown_protocols = set(arms_by_protocol) - set(self.targets_by_protocol)
            if unknown_protocols:
                raise ValueError(
                    f"arms_by_protocol has unknown protocol(s): {sorted(unknown_protocols)}"
                )
            self.arms_by_protocol = {}
            for protocol in self.targets_by_protocol:
                raw_arms = arms_by_protocol.get(protocol, self.arms)
                protocol_arms = sorted({str(arm) for arm in raw_arms})
                if not protocol_arms:
                    raise ValueError(f"protocol {protocol!r} must target at least one arm")
                unknown_arms = set(protocol_arms) - set(self.arms)
                if unknown_arms:
                    raise ValueError(
                        f"protocol {protocol!r} targets unknown arm(s): "
                        f"{sorted(unknown_arms)}; manifest arms are {self.arms}"
                    )
                self.arms_by_protocol[protocol] = protocol_arms
        for protocol, caps in self.protocol_arm_targets.items():
            unknown = set(caps) - set(self.arms_by_protocol[protocol])
            if unknown:
                raise ValueError(
                    f"arm cap [{protocol!r}] names arm(s) {sorted(unknown)} not targeted by that "
                    "protocol"
                )
        allow_duplicate_states = bool(payload.get("allow_duplicate_states", False))
        duplicate_state_resolution = payload.get("duplicate_state_resolution")
        if len(self.manifest_arr) > 1:
            min_dist = self._matcher.min_pairwise_distance()
            if min_dist <= 2.0 * self.match_tolerance and not (
                allow_duplicate_states and duplicate_state_resolution == "manifest_idx"
            ):
                raise ValueError(
                    f"{self.manifest_path}: manifest contains states within "
                    f"{min_dist:.4g} of each other, which is <= 2 * "
                    f"match_tolerance ({self.match_tolerance}). Resume and "
                    "quota matching would be ambiguous. Set "
                    "allow_duplicate_states=true and "
                    "duplicate_state_resolution='manifest_idx' only for collectors "
                    "that write and validate manifest_idx directly."
                )
        self.protocols = list(self.targets_by_protocol)
        self.counts: dict[str, Counter[str]] = {
            protocol: Counter({arm: 0 for arm in self.arms_by_protocol[protocol]})
            for protocol in self.protocols
        }
        fresh_capacity = Counter(arm for sources in self.sources_per_state for arm in sources)
        # Capacity is checked against the PROTOCOL target (a cap only lowers it, and a
        # 'collected' cap is not resolved until the ledger has been read below).
        insufficient_no_cf = {
            arm: int(fresh_capacity[arm])
            for arm in self.arms_by_protocol["no_cf"]
            if fresh_capacity[arm] < self.targets_by_protocol["no_cf"]
        }
        if insufficient_no_cf:
            raise ValueError(
                f"{self.manifest_path}: no_cf target requires at least "
                f"{self.targets_by_protocol['no_cf']} fresh starts per arm, but capacities are "
                f"{insufficient_no_cf}"
            )
        missing_fresh_source = {
            protocol: [arm for arm in arms if fresh_capacity[arm] <= 0]
            for protocol, arms in self.arms_by_protocol.items()
        }
        missing_fresh_source = {
            protocol: arms for protocol, arms in missing_fresh_source.items() if arms
        }
        if missing_fresh_source:
            raise ValueError(
                f"{self.manifest_path}: every protocol-targeted arm must appear "
                f"in at least one manifest state; missing {missing_fresh_source}"
            )
        self.fresh_consumed_manifest_idxs: set[int] = set()
        self.last_successful_manifest_idx: int | None = None
        self.last_successful_fresh_manifest_idx: int | None = None
        self.pending_retry_fresh_manifest_idx: int | None = None
        self.n_saved_rows = 0

        self.session_started_monotonic = time.monotonic()
        self.session_success_rows = 0
        self._recent_success_times: deque[float] = deque(maxlen=self.progress_window)
        self._recent_credit_units: deque[int] = deque(maxlen=self.progress_window)

        self.ledger_path.parent.mkdir(parents=True, exist_ok=True)
        self._load_existing_ledger()
        unresolved = [
            f"{protocol}.{arm}"
            for protocol, caps in self.protocol_arm_targets.items()
            for arm, target in caps.items()
            if target == "collected"
        ]
        if unresolved:
            raise ValueError(
                f"arm cap {unresolved}='collected' needs an existing ledger with rows; "
                f"{self.ledger_path} has none"
            )

    def _load_existing_ledger(self) -> None:
        if not self.ledger_path.exists():
            return
        credited_fresh_manifest_idxs: dict[int, int] = {}
        seen_episode_idxs: set[int] = set()
        for line_no, line in enumerate(self.ledger_path.read_text().splitlines(), start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            episode_index = int(row["episode_index"])
            if episode_index in seen_episode_idxs:
                raise ValueError(
                    f"{self.ledger_path}:{line_no}: duplicate episode_index {episode_index}"
                )
            seen_episode_idxs.add(episode_index)
            row_manifest_hash = row.get("manifest_hash")
            if row_manifest_hash is not None and row_manifest_hash != self.manifest_hash:
                raise ValueError(
                    f"{self.ledger_path}:{line_no}: manifest_hash mismatch; "
                    "refusing to resume a ledger from a different manifest"
                )
            # Written unconditionally by credit_episode; a missing key means a
            # malformed/foreign ledger and must fail loudly (mirrors episode_index above).
            credited = row["credited_protocol_arms"]
            if not isinstance(credited, dict):
                raise ValueError(
                    f"{self.ledger_path}:{line_no}: credited_protocol_arms must be a dict"
                )
            # Invariant (mirrors the splitter): a counterfactual
            # replay must never credit a no_cf protocol, or it would consume a no_cf
            # quota slot and later route into a no_cf actor repo.
            if bool(row["is_counterfactual"]) and credited.get("no_cf"):
                raise ValueError(
                    f"{self.ledger_path}:{line_no}: counterfactual row credits the "
                    f"no_cf protocol ({credited['no_cf']}); CF replays must never "
                    "credit a no_cf quota."
                )
            if row.get("manifest_idx") is not None:
                manifest_idx = int(row["manifest_idx"])
                if manifest_idx < 0 or manifest_idx >= len(self.states):
                    raise ValueError(
                        f"{self.ledger_path}:{line_no}: manifest_idx {manifest_idx} "
                        f"is out of range for manifest with {len(self.states)} states"
                    )
                row_sources = row.get("manifest_sources")
                if row_sources is not None and list(row_sources) != self.sources_for(manifest_idx):
                    raise ValueError(
                        f"{self.ledger_path}:{line_no}: manifest_sources {row_sources!r} "
                        f"do not match current manifest sources "
                        f"{self.sources_for(manifest_idx)!r}"
                    )
            row_was_credited = any(bool(arms) for arms in credited.values())
            for protocol, arms in credited.items():
                if protocol not in self.counts:
                    raise ValueError(f"{self.ledger_path}:{line_no}: unknown protocol {protocol!r}")
                for arm in arms:
                    if arm not in self.counts[protocol]:
                        raise ValueError(
                            f"{self.ledger_path}:{line_no}: unknown arm {arm!r} "
                            f"for protocol {protocol!r}"
                        )
                    self.counts[protocol][arm] += 1
            if row.get("manifest_idx") is not None and row_was_credited:
                self.last_successful_manifest_idx = manifest_idx
                # Hard read: a credited row without this flag is a malformed ledger;
                # silently defaulting to "fresh" would let a CF replay consume a no_cf slot.
                if not bool(row["is_counterfactual"]):
                    previous_line = credited_fresh_manifest_idxs.get(manifest_idx)
                    if previous_line is not None:
                        raise ValueError(
                            f"{self.ledger_path}:{line_no}: manifest_idx "
                            f"{manifest_idx} was already saved as a credited "
                            f"fresh episode on line {previous_line}. Duplicate "
                            "fresh credits make protocol quotas impossible to "
                            "audit safely."
                        )
                    credited_fresh_manifest_idxs[manifest_idx] = line_no
                    self.fresh_consumed_manifest_idxs.add(manifest_idx)
                    self.last_successful_fresh_manifest_idx = manifest_idx
                    if self.pending_retry_fresh_manifest_idx == manifest_idx:
                        self.pending_retry_fresh_manifest_idx = None
            elif (
                row.get("manifest_idx") is not None
                and not bool(row["is_counterfactual"])
                and not row_was_credited
            ):
                self.pending_retry_fresh_manifest_idx = manifest_idx
            self.n_saved_rows += 1

        expected_episode_idxs = list(range(self.n_saved_rows))
        actual_episode_idxs = sorted(seen_episode_idxs)
        if actual_episode_idxs != expected_episode_idxs:
            raise ValueError(
                f"{self.ledger_path}: episode_index values must be contiguous "
                f"0..{self.n_saved_rows - 1}; got {actual_episode_idxs[:20]}"
            )

        for protocol, caps in self.protocol_arm_targets.items():
            for arm, target in list(caps.items()):
                if target == "collected":
                    if self.n_saved_rows == 0:
                        raise ValueError(
                            f"arm cap {protocol}.{arm}='collected' needs an existing ledger; "
                            f"{self.ledger_path} has no rows"
                        )
                    caps[arm] = int(self.counts[protocol][arm])
        over: dict[str, dict[str, int]] = {}
        for protocol, counts in self.counts.items():
            protocol_over = {
                arm: n for arm, n in counts.items() if n > self.target_for(protocol, arm)
            }
            if protocol_over:
                over[protocol] = protocol_over
        if over:
            raise ValueError(
                f"{self.ledger_path}: existing ledger exceeds targets "
                f"{self.targets_by_protocol}: {over}"
            )

    def match(self, vec: np.ndarray | list[float] | tuple[float, ...]) -> tuple[int, float]:
        return self._matcher.query_within_tolerance(vec, self.match_tolerance)

    def sources_for(self, manifest_idx: int) -> list[str]:
        return list(self.sources_per_state[int(manifest_idx)])

    def target_for(self, protocol: str, arm: str) -> int:
        """The (protocol, arm) target: the protocol target unless the arm is capped."""
        target = self.protocol_arm_targets.get(protocol, {}).get(
            arm, self.targets_by_protocol[protocol]
        )
        if target == "collected":
            raise ValueError(
                f"arm cap {protocol}.{arm}='collected' is unresolved: the ledger "
                f"{self.ledger_path} does not exist yet"
            )
        return int(target)

    def protocol_total(self, protocol: str) -> int:
        return int(sum(self.target_for(protocol, arm) for arm in self.arms_by_protocol[protocol]))

    def protocol_count(self, protocol: str) -> int:
        return int(sum(self.counts[protocol].values()))

    def is_protocol_complete(self, protocol: str) -> bool:
        return all(
            self.counts[protocol][arm] >= self.target_for(protocol, arm)
            for arm in self.arms_by_protocol[protocol]
        )

    def is_complete(self) -> bool:
        return all(self.is_protocol_complete(protocol) for protocol in self.protocols)

    def remaining(self) -> dict[str, dict[str, int]]:
        return {
            protocol: {
                arm: max(0, self.target_for(protocol, arm) - int(count))
                for arm, count in counts.items()
            }
            for protocol, counts in self.counts.items()
        }

    def _eligible_arms_for_sources(self, protocol: str, sources: list[str]) -> list[str]:
        protocol_arms = set(self.arms_by_protocol[protocol])
        return [
            arm
            for arm in sources
            if arm in protocol_arms and self.counts[protocol][arm] < self.target_for(protocol, arm)
        ]

    def _balanced_credited_arms(
        self,
        protocol: str,
        sources: list[str],
        *,
        require_improvement_when_over_slack: bool = False,
    ) -> list[str]:
        eligible = self._eligible_arms_for_sources(protocol, sources)
        if not eligible:
            return []

        # Balance is measured over the arms still below their own target: an arm that has
        # reached its (possibly capped) target is done and must not hold the others back.
        open_arms = [
            arm
            for arm in self.arms_by_protocol[protocol]
            if self.counts[protocol][arm] < self.target_for(protocol, arm)
        ]
        if not open_arms:
            return []
        current_values = [int(self.counts[protocol][arm]) for arm in open_arms]
        current_spread = max(current_values) - min(current_values)
        projected = {
            arm: int(self.counts[protocol][arm]) + (1 if arm in eligible else 0)
            for arm in open_arms
        }
        if any(projected[arm] > self.target_for(protocol, arm) for arm in open_arms):
            return []
        projected_spread = max(projected.values()) - min(projected.values())
        if current_spread > self.balance_slack:
            # CF replays are allowed by source capacity, so they can temporarily
            # push a protocol outside the fresh-start balance slack. Fresh
            # starts for lagging arms must remain eligible so the run can
            # recover instead of dead-ending with quota remaining.
            if require_improvement_when_over_slack:
                if projected_spread >= current_spread:
                    return []
                return eligible
            if projected_spread > current_spread:
                return []
        elif projected_spread > self.balance_slack:
            return []
        return eligible

    def preview_credit(
        self,
        *,
        manifest_idx: int | None,
        success: bool,
        is_counterfactual: bool,
        enforce_fresh_balance: bool = True,
    ) -> dict[str, list[str]]:
        if not success or manifest_idx is None:
            return {}
        sources = self.sources_for(manifest_idx)
        credited: dict[str, list[str]] = {}

        if not is_counterfactual and not self.is_protocol_complete("no_cf"):
            if enforce_fresh_balance:
                no_cf = self._balanced_credited_arms("no_cf", sources)
            else:
                no_cf = self._eligible_arms_for_sources("no_cf", sources)
            if no_cf:
                credited["no_cf"] = no_cf

        if "with_cf" in self.counts and not self.is_protocol_complete("with_cf"):
            if is_counterfactual:
                with_cf = self._balanced_credited_arms(
                    "with_cf",
                    sources,
                    require_improvement_when_over_slack=True,
                )
            elif enforce_fresh_balance:
                with_cf = self._balanced_credited_arms("with_cf", sources)
            else:
                with_cf = self._eligible_arms_for_sources("with_cf", sources)
            if with_cf:
                credited["with_cf"] = with_cf

        return credited

    def _fresh_selection_protocol(self) -> str | None:
        if not self.is_protocol_complete("no_cf"):
            return "no_cf"
        if "with_cf" in self.counts and not self.is_protocol_complete("with_cf"):
            return "with_cf"
        return None

    def _fresh_protocols_for_arm(self, arm: str) -> list[str]:
        """Protocols a fresh success on ``arm`` could still credit (union mode), else the
        single no-CF-first selection protocol."""
        if not self.fresh_union_selection:
            protocol = self._fresh_selection_protocol()
            return [] if protocol is None else [protocol]
        return [
            protocol
            for protocol in self.protocols
            if arm in self.arms_by_protocol[protocol] and self.remaining()[protocol][arm] > 0
        ]

    def _fresh_credit_for_selection(
        self,
        *,
        manifest_idx: int,
        enforce_balance: bool,
    ) -> dict[str, list[str]]:
        return self.preview_credit(
            manifest_idx=manifest_idx,
            success=True,
            is_counterfactual=False,
            enforce_fresh_balance=enforce_balance,
        )

    def next_fresh_manifest_idx_for_arm(
        self,
        arm: str,
        *,
        protocol: str | None = None,
        enforce_balance: bool | None = None,
    ) -> int | None:
        if protocol is None:
            protocols = self._fresh_protocols_for_arm(arm)
            if not protocols:
                return None
        else:
            protocols = [str(protocol)]
            if protocols[0] not in self.counts:
                raise ValueError(f"unknown protocol {protocols[0]!r}")
            if arm not in self.arms_by_protocol[protocols[0]]:
                raise ValueError(f"arm {arm!r} is not targeted by protocol {protocols[0]!r}")
            if self.remaining()[protocols[0]][arm] <= 0:
                return None
        if enforce_balance is None:
            enforce_balance = self.selection_mode == "hard_balance"

        for manifest_idx, sources in enumerate(self.sources_per_state):
            if manifest_idx in self.fresh_consumed_manifest_idxs:
                continue
            if arm not in sources:
                continue
            credited = self._fresh_credit_for_selection(
                manifest_idx=manifest_idx,
                enforce_balance=enforce_balance,
            )
            if any(arm in credited.get(p, []) for p in protocols):
                return manifest_idx
        return None

    def soft_weighted_fresh_arm_probabilities(self) -> dict[str, float]:
        if self._fresh_selection_protocol() is None:
            return {}
        remaining_all = self.remaining()
        candidates: list[str] = []
        candidate_remaining: list[int] = []
        for arm in self.arms:
            protocols = self._fresh_protocols_for_arm(arm)
            if not protocols:
                continue
            # union mode weights an arm by the most it still owes on any protocol
            arm_remaining = max(int(remaining_all[p][arm]) for p in protocols)
            if arm_remaining <= 0:
                continue
            if self.next_fresh_manifest_idx_for_arm(arm, enforce_balance=False) is None:
                continue
            candidates.append(arm)
            candidate_remaining.append(arm_remaining)
        if not candidates:
            return {}

        values = np.asarray(candidate_remaining, dtype=np.float64)
        logits = self.softmax_beta * (values - float(values.min()))
        logits = np.clip(logits, a_min=None, a_max=700.0)
        logits -= float(logits.max())
        weights = np.exp(logits)
        probs = weights / float(weights.sum())
        return {arm: float(prob) for arm, prob in zip(candidates, probs, strict=True)}

    def _soft_selection_seed(self) -> int:
        payload = {
            "sampling_seed": self.sampling_seed,
            "manifest_hash": self.manifest_hash,
            "n_saved_rows": self.n_saved_rows,
            "counts": {
                protocol: dict(sorted(counts.items()))
                for protocol, counts in sorted(self.counts.items())
            },
            "fresh_consumed_manifest_idxs": sorted(self.fresh_consumed_manifest_idxs),
        }
        digest = hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).digest()
        return int.from_bytes(digest[:8], "little", signed=False)

    def select_fresh_manifest_idx(self) -> int | None:
        if self.is_complete():
            return None

        if self.pending_retry_fresh_manifest_idx is not None and self.is_fresh_state_eligible(
            self.pending_retry_fresh_manifest_idx,
            enforce_balance=False,
        ):
            return self.pending_retry_fresh_manifest_idx

        if self.selection_mode == "hard_balance":
            for manifest_idx in range(len(self.states)):
                if self.is_fresh_state_eligible(manifest_idx):
                    return manifest_idx
            return None

        probabilities = self.soft_weighted_fresh_arm_probabilities()
        if not probabilities:
            return None
        arms = list(probabilities)
        probs = np.asarray([probabilities[arm] for arm in arms], dtype=np.float64)
        rng = np.random.default_rng(self._soft_selection_seed())
        selected_arm = str(rng.choice(arms, p=probs))
        manifest_idx = self.next_fresh_manifest_idx_for_arm(selected_arm, enforce_balance=False)
        if manifest_idx is None:
            raise RuntimeError(
                "soft-weighted selection chose an arm with no unconsumed fresh "
                f"manifest row: arm={selected_arm!r}, remaining={self.remaining()}"
            )
        return manifest_idx

    def is_fresh_state_eligible(
        self,
        manifest_idx: int,
        *,
        enforce_balance: bool | None = None,
    ) -> bool:
        manifest_idx = int(manifest_idx)
        if manifest_idx in self.fresh_consumed_manifest_idxs:
            return False
        if self.is_complete():
            return False
        if enforce_balance is None:
            enforce_balance = self.selection_mode == "hard_balance"

        sources = self.sources_for(manifest_idx)
        no_cf = []
        if not self.is_protocol_complete("no_cf"):
            if enforce_balance:
                no_cf = self._balanced_credited_arms("no_cf", sources)
            else:
                no_cf = self._eligible_arms_for_sources("no_cf", sources)

        with_cf = []
        if "with_cf" in self.counts and not self.is_protocol_complete("with_cf"):
            if enforce_balance:
                with_cf = self._balanced_credited_arms("with_cf", sources)
            else:
                with_cf = self._eligible_arms_for_sources("with_cf", sources)

        if self.fresh_union_selection:
            # Per-arm caps: an arm whose no-CF is capped may still owe with-CF, so a fresh
            # start is eligible when its success credits ANY open protocol.
            return bool(no_cf) or bool(with_cf)

        if not self.is_protocol_complete("no_cf"):
            # A fresh successful rollout necessarily consumes a no-CF start, so
            # no-CF balance is the hard eligibility gate while no-CF remains.
            # With-CF can temporarily need a different arm than no-CF near the
            # tail after counterfactual replays; requiring the same fresh state
            # to satisfy both protocols can exhaust the sampler with valid
            # no-CF starts still available.
            return bool(no_cf)

        if "with_cf" in self.counts and not self.is_protocol_complete("with_cf"):
            return bool(with_cf)
        return False

    def can_accept_counterfactual(self, manifest_idx: int | None) -> bool:
        if manifest_idx is None or "with_cf" not in self.counts:
            return False
        if self.is_protocol_complete("with_cf"):
            return False
        return bool(
            self.preview_credit(
                manifest_idx=manifest_idx,
                success=True,
                is_counterfactual=True,
            ).get("with_cf")
        )

    def resume_counterfactual_manifest_idx(self) -> int | None:
        """Return the latest credited state that can still accept CF."""
        manifest_idx = self.last_successful_manifest_idx
        if manifest_idx is None:
            return None
        if not self.can_accept_counterfactual(manifest_idx):
            return None
        return manifest_idx

    def state_score(self, manifest_idx: int) -> int:
        credited = self.preview_credit(
            manifest_idx=manifest_idx,
            success=True,
            is_counterfactual=False,
        )
        remaining = self.remaining()
        no_cf_score = sum(
            remaining.get("no_cf", {}).get(arm, 0) for arm in credited.get("no_cf", [])
        )
        with_cf_score = sum(
            remaining.get("with_cf", {}).get(arm, 0) for arm in credited.get("with_cf", [])
        )
        # With-CF is the protocol that can be perturbed by optional CF replays,
        # so while it is open, fresh-state selection must resolve its arm
        # balance first and use no-CF as the secondary tiebreaker. This avoids a
        # summed score letting the larger no-CF tail swamp a small with-CF
        # imbalance near the end of collection.
        if "with_cf" in self.counts and not self.is_protocol_complete("with_cf"):
            multiplier = self.protocol_total("no_cf") + self.protocol_total("with_cf") + 1
            return with_cf_score * multiplier + no_cf_score
        return no_cf_score

    def credit_episode(
        self,
        *,
        manifest_idx: int | None,
        episode_index: int,
        success: bool,
        is_counterfactual: bool,
        matched_distance: float | None = None,
        extra: dict[str, Any] | None = None,
        quota_credit: bool | None = None,
        write_ledger: bool = True,
    ) -> dict[str, Any]:
        should_credit = bool(success) if quota_credit is None else bool(quota_credit)
        if manifest_idx is not None:
            manifest_idx = int(manifest_idx)
        if (
            should_credit
            and manifest_idx is not None
            and not is_counterfactual
            and manifest_idx in self.fresh_consumed_manifest_idxs
        ):
            raise ValueError(
                f"manifest_idx {manifest_idx} was already credited as a "
                "fresh episode. Refusing to double-count no-CF quota."
            )

        credited = self.preview_credit(
            manifest_idx=manifest_idx,
            success=should_credit,
            is_counterfactual=is_counterfactual,
            enforce_fresh_balance=is_counterfactual,
        )
        if should_credit and manifest_idx is not None and not credited and not self.is_complete():
            raise ValueError(
                f"episode for manifest_idx {manifest_idx} would "
                f"not credit any protocol quota; remaining={self.remaining()}"
            )
        for protocol, arms in credited.items():
            for arm in arms:
                self.counts[protocol][arm] += 1

        if manifest_idx is None:
            sources: list[str] = []
        else:
            sources = self.sources_for(manifest_idx)
            if should_credit:
                self.last_successful_manifest_idx = manifest_idx
                if not is_counterfactual:
                    self.fresh_consumed_manifest_idxs.add(manifest_idx)
                    self.last_successful_fresh_manifest_idx = manifest_idx
                    if self.pending_retry_fresh_manifest_idx == manifest_idx:
                        self.pending_retry_fresh_manifest_idx = None
            elif not is_counterfactual:
                self.pending_retry_fresh_manifest_idx = manifest_idx

        row = {
            "episode_index": int(episode_index),
            "success": bool(success),
            "quota_credit": bool(should_credit),
            "is_counterfactual": bool(is_counterfactual),
            "manifest_idx": None if manifest_idx is None else int(manifest_idx),
            "manifest_sources": sources,
            "credited_protocol_arms": credited,
            "counts_after": {
                protocol: {
                    arm: int(self.counts[protocol][arm]) for arm in self.arms_by_protocol[protocol]
                }
                for protocol in self.protocols
            },
            "remaining_after": self.remaining(),
            "matched_distance": matched_distance,
            "manifest_hash": self.manifest_hash,
            "written_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }
        if extra:
            row.update(extra)

        if write_ledger:
            self.append_reserved_row(row)
        if should_credit:
            self.session_success_rows += 1
            self._recent_success_times.append(time.monotonic())
            self._recent_credit_units.append(sum(len(arms) for arms in credited.values()))
        return row

    def append_reserved_row(self, row: dict[str, Any]) -> None:
        """Write a pre-credited row after its dataset save has completed."""
        with self.ledger_path.open("a") as f:
            f.write(json.dumps(row, sort_keys=True) + "\n")
        self.n_saved_rows += 1

    def _format_duration(self, seconds: float | None) -> str:
        if seconds is None or not np.isfinite(seconds) or seconds < 0:
            return "unknown"
        seconds = int(round(seconds))
        hours, rem = divmod(seconds, 3600)
        minutes, secs = divmod(rem, 60)
        if hours:
            return f"{hours}h {minutes:02d}m"
        if minutes:
            return f"{minutes}m {secs:02d}s"
        return f"{secs}s"

    def _recent_seconds_per_success(self) -> float | None:
        if not self._recent_success_times:
            return None
        if len(self._recent_success_times) >= 2:
            elapsed = self._recent_success_times[-1] - self._recent_success_times[0]
            return elapsed / max(1, len(self._recent_success_times) - 1)
        elapsed = time.monotonic() - self.session_started_monotonic
        return elapsed / max(1, self.session_success_rows)

    def _recent_credits_per_success(self, protocol: str | None = None) -> float | None:
        if not self._recent_credit_units:
            return None
        if protocol is None:
            return float(np.mean(self._recent_credit_units))
        # Conservative protocol-specific fallback: use aggregate credit rate
        # scaled by the protocol's share of remaining work.
        return float(np.mean(self._recent_credit_units))

    def _eta_for_remaining_units(self, remaining_units: int) -> str:
        if remaining_units <= 0:
            return "complete"
        sec_per_success = self._recent_seconds_per_success()
        credits_per_success = self._recent_credits_per_success()
        if sec_per_success is None or credits_per_success is None or credits_per_success <= 0:
            return "unknown"
        estimated_successes = remaining_units / credits_per_success
        return self._format_duration(estimated_successes * sec_per_success)

    def _estimated_successes_for_remaining_units(self, remaining_units: int) -> str:
        if remaining_units <= 0:
            return "0"
        credits_per_success = self._recent_credits_per_success()
        if credits_per_success is None or credits_per_success <= 0:
            return "unknown"
        return str(int(np.ceil(remaining_units / credits_per_success)))

    def progress_lines(
        self,
        *,
        saved_episode_count: int,
        total_manifest_remaining: int | None = None,
        eligible_manifest_remaining: int | None = None,
    ) -> list[str]:
        lines = ["Progress"]
        for protocol in self.protocols:
            count = self.protocol_count(protocol)
            total = self.protocol_total(protocol)
            status = "complete" if count >= total else f"{count} / {total} credits"
            label = "With-CF quota" if protocol == "with_cf" else "No-CF quota"
            lines.append(f"  {label}: {status}")

        if total_manifest_remaining is not None and eligible_manifest_remaining is not None:
            lines.append(
                "  Fresh starts in manifest: "
                f"{total_manifest_remaining} total remaining, "
                f"{eligible_manifest_remaining} quota-eligible"
            )
        lines.append(
            f"  Saved this session: {saved_episode_count} episodes, "
            f"{self.session_success_rows} credited"
        )

        sec_per_success = self._recent_seconds_per_success()
        if sec_per_success is None:
            lines.append("  Pace: unknown")
        else:
            lines.append(
                "  Pace: "
                f"{self._format_duration(sec_per_success)}/credited episode "
                f"over last {len(self._recent_success_times)} credited episode(s)"
            )

        free_remaining = 0
        if "with_cf" in self.counts:
            free_remaining = sum(self.remaining()["with_cf"].values())
            lines.append(f"  ETA to With-CF full: {self._eta_for_remaining_units(free_remaining)}")

        total_remaining = sum(
            sum(protocol_remaining.values()) for protocol_remaining in self.remaining().values()
        )
        lines.append(
            "  Estimated remaining credited episodes: "
            f"{self._estimated_successes_for_remaining_units(total_remaining)}"
        )
        lines.append(f"  ETA to all quotas full: {self._eta_for_remaining_units(total_remaining)}")
        if (
            "with_cf" in self.counts
            and self.is_protocol_complete("with_cf")
            and not self.is_complete()
        ):
            lines.append("  Mode: normal rollouts only")
        return lines

    def operator_panel_status(self) -> dict[str, Any]:
        """Quota / pace / ETA for the operator panel header (``CollectionStatus`` fields).

        ``metrics``: one ``(LABEL, value)`` cell per protocol plus the ETA to all quotas
        full; ``fraction``: credited units over targeted units; ``summary``: credits and
        pace this session. Distinct from :meth:`progress_lines`, the verbose terminal
        readout printed after each episode.
        """
        metrics = []
        credited = targeted = 0
        for protocol in self.protocols:
            label = "WITH-CF QUOTA" if protocol == "with_cf" else "NO-CF QUOTA"
            count, total = self.protocol_count(protocol), self.protocol_total(protocol)
            metrics.append((label, f"{count} / {total}"))
            credited += min(count, total)
            targeted += total
        total_remaining = sum(sum(remaining.values()) for remaining in self.remaining().values())
        metrics.append(("EST. TIME LEFT", self._eta_for_remaining_units(total_remaining)))
        sec_per_success = self._recent_seconds_per_success()
        pace = (
            "gathering"
            if sec_per_success is None
            else f"{self._format_duration(sec_per_success)} / ep"
        )
        summary = (
            f"Credited {self.session_success_rows}   |   {pace}   |   "
            f"~{self._estimated_successes_for_remaining_units(total_remaining)} left"
        )
        return {
            "metrics": tuple(metrics),
            "fraction": credited / targeted,
            "summary": summary,
        }


def vector_from_narrow_state(state: tuple[float, float, float]) -> np.ndarray:
    x, y, yaw = state
    return np.array([x, y, yaw], dtype=np.float64)


def vector_from_broad_state(
    state: tuple[tuple[float, float, float], tuple[float, float]],
) -> np.ndarray:
    nut_state, peg_state = state
    return np.array(
        [nut_state[0], nut_state[1], nut_state[2], peg_state[0], peg_state[1]],
        dtype=np.float64,
    )

from dataclasses import dataclass

from convex import ConvexInt64


@dataclass
class PolicyInput:
    name: str
    model_id: str
    environment: str
    model_url: str | None = None
    training_url: str | None = None

    def to_dict(self) -> dict:
        d = {
            "name": self.name,
            "model_id": self.model_id,
            "environment": self.environment,
        }
        if self.model_url is not None:
            d["model_url"] = self.model_url
        if self.training_url is not None:
            d["training_url"] = self.training_url
        return d


@dataclass
class RoundResultInput:
    model_id: str
    success: bool
    episode_index: int
    num_frames: int | None = None
    # Mid-episode sub-goal marks reached (graded score = marks + success).
    num_subtask_marks: int | None = None

    def to_dict(self) -> dict:
        d = {
            "model_id": self.model_id,
            "success": self.success,
            "episode_index": ConvexInt64(self.episode_index),
        }
        if self.num_frames is not None:
            d["num_frames"] = ConvexInt64(self.num_frames)
        if self.num_subtask_marks is not None:
            d["num_subtask_marks"] = ConvexInt64(self.num_subtask_marks)
        return d


@dataclass
class RoundInput:
    round_index: int
    results: list[RoundResultInput]

    def to_dict(self) -> dict:
        return {
            "round_index": ConvexInt64(self.round_index),
            "results": [r.to_dict() for r in self.results],
        }


@dataclass
class DatasetInput:
    repo_id: str
    name: str
    task: str
    source_type: str  # provenance: "teleop" | "rollout" | "dagger" | "eval"
    environment: str
    num_episodes: int | None = None
    model_id: str | None = None
    model_url: str | None = None
    dataset_role: str | None = None
    trainable: bool | None = None
    parent_repo_id: str | None = None
    derived_repo_ids: list[str] | None = None
    mutually_exclusive_with: list[str] | None = None
    view_family_id: str | None = None
    view_id: str | None = None
    producer_model_ids: list[str] | None = None
    target_model_id: str | None = None
    target_arm_key: str | None = None
    notes: str | None = None

    def to_dict(self) -> dict:
        d = {
            "repo_id": self.repo_id,
            "name": self.name,
            "task": self.task,
            "source_type": self.source_type,
            "environment": self.environment,
        }
        if self.num_episodes is not None:
            d["num_episodes"] = ConvexInt64(self.num_episodes)
        if self.model_id is not None:
            d["model_id"] = self.model_id
        if self.model_url is not None:
            d["model_url"] = self.model_url
        if self.dataset_role is not None:
            d["dataset_role"] = self.dataset_role
        if self.trainable is not None:
            d["trainable"] = self.trainable
        if self.parent_repo_id is not None:
            d["parent_repo_id"] = self.parent_repo_id
        if self.derived_repo_ids is not None:
            d["derived_repo_ids"] = self.derived_repo_ids
        if self.mutually_exclusive_with is not None:
            d["mutually_exclusive_with"] = self.mutually_exclusive_with
        if self.view_family_id is not None:
            d["view_family_id"] = self.view_family_id
        if self.view_id is not None:
            d["view_id"] = self.view_id
        if self.producer_model_ids is not None:
            d["producer_model_ids"] = self.producer_model_ids
        if self.target_model_id is not None:
            d["target_model_id"] = self.target_model_id
        if self.target_arm_key is not None:
            d["target_arm_key"] = self.target_arm_key
        if self.notes is not None:
            d["notes"] = self.notes
        return d

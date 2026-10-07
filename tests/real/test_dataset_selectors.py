import pandas as pd
import pytest

from mulligan.real.train.dataset_selectors import (
    EpisodeSelector,
    parse_dataset_episodes,
    parse_dataset_revisions,
    parse_episode_selector,
    resolve_episode_indices,
    selectors_to_json,
    validate_against_repo_ids,
)


def test_parse_revisions():
    assert parse_dataset_revisions(["a/b=123", "c/d=456"]) == {"a/b": "123", "c/d": "456"}
    assert parse_dataset_revisions(None) == {}
    with pytest.raises(ValueError, match="Conflicting"):
        parse_dataset_revisions(["a/b=1", "a/b=2"])
    with pytest.raises(ValueError, match="<repo>=<value>"):
        parse_dataset_revisions(["a/b"])


def test_parse_selectors_round_trip():
    sel = parse_episode_selector("episode_index:0-3,7,9-10")
    assert sel == EpisodeSelector("episode_index", (0, 1, 2, 3, 7, 9, 10))
    assert sel.to_cli() == "episode_index:0-3,7,9-10"
    assert parse_episode_selector(sel.to_cli()) == sel
    sess = parse_episode_selector("session_id:b01,b03")
    assert sess.values == ("b01", "b03")
    assert EpisodeSelector.from_json(sess.to_json()) == sess
    parsed = parse_dataset_episodes(["x/y=session_id:b03", "z/w=episode_index:4"])
    assert (
        selectors_to_json(parsed)
        == '{"x/y": {"session_id": ["b03"]}, "z/w": {"episode_index": [4]}}'
    )


@pytest.mark.parametrize("text", ["b03", "sessions:b03", "episode_index:", "episode_index:5-2"])
def test_bad_selectors(text):
    with pytest.raises(ValueError):
        parse_episode_selector(text)


def test_validate_against_repo_ids():
    sel = {"a/b": parse_episode_selector("session_id:b01")}
    validate_against_repo_ids(["a/b"], {"a/b": "rev"}, sel)
    with pytest.raises(ValueError, match="not in the run's repo ids"):
        validate_against_repo_ids(["c/d"], {"a/b": "rev"}, {})
    with pytest.raises(ValueError, match="pinned --dataset-revisions"):
        validate_against_repo_ids(["a/b"], {}, sel)


def test_resolve_session_selector_from_provenance():
    provenance = pd.DataFrame(
        {"episode_index": [0, 1, 2, 3, 4], "session_id": ["b01", "b01", "b02", "b03", "b03"]}
    )
    sel = parse_episode_selector("session_id:b03,b01")
    assert resolve_episode_indices("x/y", sel, "rev", provenance=provenance) == [0, 1, 3, 4]
    with pytest.raises(ValueError, match="sessions"):
        resolve_episode_indices(
            "x/y", parse_episode_selector("session_id:b09"), "rev", provenance=provenance
        )
    idx = parse_episode_selector("episode_index:2-3")
    assert resolve_episode_indices("x/y", idx, "rev") == [2, 3]

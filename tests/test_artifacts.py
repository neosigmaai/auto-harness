"""Unit tests for the local artifact store (no DB, no harbor, no network)."""

from __future__ import annotations

from pathlib import Path

import pytest

from api.config import REPO_ROOT, BenchmarkConfig
from api.services.artifacts import (
    LocalArtifactStore,
    create_artifact_store,
    improver_key,
    result_key,
    trace_key,
)


def test_put_and_get_bytes_round_trip(tmp_path: Path) -> None:
    store = LocalArtifactStore(tmp_path)
    store.put("jobs/j1/iterations/0/tasks/fix-git/trace.json", b'{"a": 1}')
    assert store.get("jobs/j1/iterations/0/tasks/fix-git/trace.json") == b'{"a": 1}'


def test_put_and_get_str_round_trip(tmp_path: Path) -> None:
    store = LocalArtifactStore(tmp_path)
    store.put("jobs/j1/iterations/0/improver/prompt.txt", "héllo prompt")
    assert store.get("jobs/j1/iterations/0/improver/prompt.txt") == "héllo prompt".encode()


def test_put_copies_a_file_given_as_path(tmp_path: Path) -> None:
    source = tmp_path / "harbor_out" / "trace.json"
    source.parent.mkdir(parents=True)
    source.write_bytes(b"[]")
    store = LocalArtifactStore(tmp_path / "store")

    store.put("jobs/j1/iterations/0/tasks/fix-git/trace.json", source)

    assert store.get("jobs/j1/iterations/0/tasks/fix-git/trace.json") == b"[]"
    assert source.exists(), "put must copy, not move, the source file"


def test_put_creates_intermediate_directories(tmp_path: Path) -> None:
    store = LocalArtifactStore(tmp_path / "nope" / "deeper")
    store.put("a/b/c/d.json", b"1")
    assert (tmp_path / "nope" / "deeper" / "a" / "b" / "c" / "d.json").is_file()


def test_put_overwrites_an_existing_key(tmp_path: Path) -> None:
    store = LocalArtifactStore(tmp_path)
    store.put("k.txt", b"first")
    store.put("k.txt", b"second")
    assert store.get("k.txt") == b"second"


def test_exists(tmp_path: Path) -> None:
    store = LocalArtifactStore(tmp_path)
    assert store.exists("jobs/j1/x.json") is False
    store.put("jobs/j1/x.json", b"{}")
    assert store.exists("jobs/j1/x.json") is True


def test_exists_is_false_for_a_directory(tmp_path: Path) -> None:
    store = LocalArtifactStore(tmp_path)
    store.put("jobs/j1/x.json", b"{}")
    assert store.exists("jobs/j1") is False


def test_get_missing_key_raises_file_not_found(tmp_path: Path) -> None:
    store = LocalArtifactStore(tmp_path)
    with pytest.raises(FileNotFoundError):
        store.get("jobs/j1/missing.json")


def test_list_by_prefix_returns_sorted_relative_keys(tmp_path: Path) -> None:
    store = LocalArtifactStore(tmp_path)
    store.put("jobs/j1/iterations/0/tasks/b/trace.json", b"1")
    store.put("jobs/j1/iterations/0/tasks/a/trace.json", b"1")
    store.put("jobs/j1/iterations/1/tasks/a/trace.json", b"1")
    store.put("jobs/j2/iterations/0/tasks/a/trace.json", b"1")

    assert store.list("jobs/j1/iterations/0") == [
        "jobs/j1/iterations/0/tasks/a/trace.json",
        "jobs/j1/iterations/0/tasks/b/trace.json",
    ]
    assert len(store.list("jobs/j1")) == 3
    assert len(store.list("")) == 4


def test_list_matches_partial_segments(tmp_path: Path) -> None:
    store = LocalArtifactStore(tmp_path)
    store.put("jobs/j1/a.json", b"1")
    store.put("jobs/j12/a.json", b"1")
    assert store.list("jobs/j1") == ["jobs/j1/a.json", "jobs/j12/a.json"]
    assert store.list("jobs/j1/") == ["jobs/j1/a.json"]


def test_list_of_missing_prefix_is_empty(tmp_path: Path) -> None:
    store = LocalArtifactStore(tmp_path)
    assert store.list("jobs/nope") == []


@pytest.mark.parametrize(
    "bad_key",
    [
        "jobs/../../etc/passwd",
        "../escape.json",
        "..",
        "/absolute/path.json",
        "jobs\\j1\\trace.json",
        "",
        "jobs/j1/",
    ],
)
def test_unsafe_keys_are_rejected(tmp_path: Path, bad_key: str) -> None:
    store = LocalArtifactStore(tmp_path)
    with pytest.raises(ValueError):
        store.put(bad_key, b"pwned")
    with pytest.raises(ValueError):
        store.get(bad_key)
    with pytest.raises(ValueError):
        store.exists(bad_key)


def test_traversal_prefix_is_rejected_by_list(tmp_path: Path) -> None:
    store = LocalArtifactStore(tmp_path)
    with pytest.raises(ValueError):
        store.list("../")


def test_key_helper_formats() -> None:
    assert (
        trace_key("j1", 0, "fix-git")
        == "jobs/j1/iterations/0/tasks/fix-git/trace.json"
    )
    assert (
        result_key("j1", 3, "regex-log")
        == "jobs/j1/iterations/3/tasks/regex-log/result.json"
    )
    assert improver_key("j1", 2, "prompt.txt") == "jobs/j1/iterations/2/improver/prompt.txt"
    assert improver_key("j1", 2, "response.json") == "jobs/j1/iterations/2/improver/response.json"


def test_create_artifact_store_resolves_relative_dir_against_repo_root() -> None:
    cfg = BenchmarkConfig(
        default_task_ids=["fix-git"],
        default_agent_model="gpt-4.1-mini",
        artifacts_dir="workspace/artifacts",
    )
    store = create_artifact_store(cfg)
    assert isinstance(store, LocalArtifactStore)
    assert store.root == REPO_ROOT / "workspace" / "artifacts"


def test_create_artifact_store_honours_an_absolute_dir(tmp_path: Path) -> None:
    cfg = BenchmarkConfig(
        default_task_ids=["fix-git"],
        default_agent_model="gpt-4.1-mini",
        artifacts_dir=str(tmp_path / "arts"),
    )
    store = create_artifact_store(cfg)
    assert isinstance(store, LocalArtifactStore)
    assert store.root == tmp_path / "arts"


def test_create_artifact_store_does_not_touch_the_filesystem(tmp_path: Path) -> None:
    """The factory is cheap: directories appear on first put, not on construction."""
    cfg = BenchmarkConfig(
        default_task_ids=["fix-git"],
        default_agent_model="gpt-4.1-mini",
        artifacts_dir=str(tmp_path / "lazy"),
    )
    create_artifact_store(cfg)
    assert not (tmp_path / "lazy").exists()

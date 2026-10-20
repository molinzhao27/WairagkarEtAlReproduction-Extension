from pathlib import Path

from wvx.config import REPO_ROOT, deep_merge, load_config


def test_default_config_loads_with_seed_42():
    cfg = load_config()
    assert cfg["seed"] == 42


def test_paths_resolve_inside_repo():
    cfg = load_config()
    for name, p in cfg["paths"].items():
        assert isinstance(p, Path), name
        assert p.is_relative_to(REPO_ROOT), f"{name} points outside the repo: {p}"


def test_deep_merge_overrides_nested_keys_without_mutating_base():
    base = {"a": {"b": 1, "c": 2}, "d": 3}
    merged = deep_merge(base, {"a": {"b": 10}})
    assert merged == {"a": {"b": 10, "c": 2}, "d": 3}
    assert base["a"]["b"] == 1

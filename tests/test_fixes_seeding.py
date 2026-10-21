"""AGENTS.md rule 7 (seeds everywhere) and result provenance for the pipeline scripts (2026-10-21 audit).
Static checks on the source: every listed script calls wvx.seeding.set_seed, and every one that writes a JSON
summary records code_version() and config_sha256(...) in it."""

from __future__ import annotations

import ast

import pytest

from wvx.config import REPO_ROOT

SEEDED = ["check_env", "prepare_kunz", "extract_kunz_subset", "inspect_mat", "fetch_third_party",
          "fetch_wairagkar_data", "time_training", "train", "align_targets", "validate_alignment", "make_tts",
          "compare_features", "asr_eval", "run_inference", "compare_closed_loop"]
# Scripts that write a JSON summary themselves (train.py's outputs are written by wvx.training).
JSON_WRITERS = ["inspect_mat", "time_training", "align_targets", "validate_alignment", "make_tts", "compare_features",
                "asr_eval", "run_inference", "compare_closed_loop"]


def _calls(name: str) -> set:
    tree = ast.parse((REPO_ROOT / "scripts" / f"{name}.py").read_text())
    out = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            f = node.func
            out.add(f.id if isinstance(f, ast.Name) else f.attr if isinstance(f, ast.Attribute) else None)
    return out


def _imports_from(name: str, module: str) -> set:
    tree = ast.parse((REPO_ROOT / "scripts" / f"{name}.py").read_text())
    return {a.name for n in ast.walk(tree) if isinstance(n, ast.ImportFrom) and n.module == module for a in n.names}


@pytest.mark.parametrize("name", SEEDED)
def test_script_calls_set_seed(name):
    assert "set_seed" in _calls(name)
    assert "set_seed" in _imports_from(name, "wvx.seeding")


@pytest.mark.parametrize("name", JSON_WRITERS)
def test_json_summary_records_code_version_and_config_hash(name):
    calls = _calls(name)
    assert {"code_version", "config_sha256"} <= calls
    assert {"code_version", "config_sha256"} <= _imports_from(name, "wvx.provenance")


def test_check_env_seed_comes_from_the_config():
    src = (REPO_ROOT / "scripts" / "check_env.py").read_text()
    assert "SEED = 42" not in src and 'load_config()["seed"]' in src

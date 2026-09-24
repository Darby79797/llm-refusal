"""Tests for config-plumbing fixes: --json defaults merge, pre-split data
shape detection, and ConceptDefinition's search_config default.

Pure config/data-shape tests — no model is loaded.
"""
import copy
import json

import pytest

from run_experiment import build_default_config, apply_config_defaults, parse_args
from framework import normalize_train_data_result
from concept import ConceptDefinition, DEFAULT_SEARCH_CONFIG


# --- FIX 1: --json defaults merge ---

def test_minimal_json_config_gets_all_defaults():
    """The docs' own minimal example must produce a fully-populated config."""
    defaults = build_default_config()
    config = apply_config_defaults({"model_name": "x", "mode": "search"})

    assert config["model_name"] == "x"
    assert config["mode"] == "search"
    assert config["torch_dtype"] == "auto"
    assert config["force_cpu"] is False
    assert config["filter_prompts"] is True
    assert config["concept"] == "refusal"

    # Every argparse-derived default key must be present.
    for key in defaults:
        assert key in config, f"missing default key: {key}"


def test_minimal_json_config_via_parse_args():
    """End-to-end through parse_args() (no model loading occurs here)."""
    config = parse_args(["--json", json.dumps({"model_name": "x", "mode": "search"})])
    assert config["model_name"] == "x"
    assert config["mode"] == "search"
    assert config["torch_dtype"] == "auto"
    assert config["force_cpu"] is False
    assert config["filter_prompts"] is True
    assert config["concept"] == "refusal"
    assert config["induce_mode"] == "single_layer"
    assert config["alpaca_max_prompts"] == 500


def test_unknown_json_key_raises():
    with pytest.raises(ValueError, match="Unknown config key"):
        apply_config_defaults({"model_name": "x", "mode": "search", "bogus_typo_key": 1})


def test_unknown_json_key_via_parse_args_exits():
    with pytest.raises(SystemExit):
        parse_args(["--json", json.dumps({"model_name": "x", "mode": "search", "bogus_typo_key": 1})])


def test_json_override_wins_over_default():
    config = apply_config_defaults({"model_name": "x", "mode": "search", "filter_prompts": False})
    assert config["filter_prompts"] is False

    config2 = apply_config_defaults({"model_name": "x", "mode": "search", "concept": "sycophancy"})
    assert config2["concept"] == "sycophancy"


# --- FIX 2: pre-split data shape detection ---

def test_normalize_simple_shape_lists():
    result = (["p1", "p2"], ["n1", "n2"])
    is_presplit, normalized = normalize_train_data_result(result, "dummy")
    assert is_presplit is False
    assert normalized == (["p1", "p2"], ["n1", "n2"])


def test_normalize_presplit_shape_lists():
    result = ((["tp1"], ["tn1"]), (["vp1"], ["vn1"]))
    is_presplit, normalized = normalize_train_data_result(result, "dummy")
    assert is_presplit is True
    assert normalized == ((["tp1"], ["tn1"]), (["vp1"], ["vn1"]))


def test_normalize_presplit_shape_tuples():
    """Tuples instead of lists for the innermost prompt collections must also be detected."""
    result = ((("tp1", "tp2"), ("tn1",)), (("vp1",), ("vn1", "vn2")))
    is_presplit, normalized = normalize_train_data_result(result, "dummy")
    assert is_presplit is True
    (train_pos, train_neg), (val_pos, val_neg) = normalized
    assert train_pos == ["tp1", "tp2"]
    assert train_neg == ["tn1"]
    assert val_pos == ["vp1"]
    assert val_neg == ["vn1", "vn2"]


def test_normalize_malformed_shape_raises():
    with pytest.raises(ValueError, match="unrecognized shape"):
        normalize_train_data_result({"positive": ["p1"], "negative": ["n1"]}, "my_concept")


def test_normalize_malformed_shape_names_concept():
    with pytest.raises(ValueError, match="my_concept"):
        normalize_train_data_result(12345, "my_concept")


def test_normalize_mixed_nested_non_strings_raises():
    """Nested but non-string elements shouldn't be silently treated as pre-split."""
    result = ((["tp1"], ["tn1"]), (1, 2))
    with pytest.raises(ValueError):
        normalize_train_data_result(result, "dummy")


# --- FIX 3: ConceptDefinition default search_config ---

def test_concept_definition_default_search_config_matches_module_default():
    concept = ConceptDefinition(
        name="test_concept",
        train_data_fn=lambda: (["p"], ["n"]),
        eval_data_fn=lambda: (["p"], ["n"]),
        target_tokens=["I"],
        detection_phrases=["I cannot"],
    )
    assert concept.search_config == DEFAULT_SEARCH_CONFIG


def test_concept_definition_default_search_config_is_independent_copy():
    concept = ConceptDefinition(
        name="test_concept",
        train_data_fn=lambda: (["p"], ["n"]),
        eval_data_fn=lambda: (["p"], ["n"]),
        target_tokens=["I"],
        detection_phrases=["I cannot"],
    )
    original = copy.deepcopy(DEFAULT_SEARCH_CONFIG)
    concept.search_config["layer_cutoff_frac"] = 0.1234
    assert DEFAULT_SEARCH_CONFIG == original

    # A second concept built without an explicit search_config gets a fresh copy too.
    concept2 = ConceptDefinition(
        name="test_concept_2",
        train_data_fn=lambda: (["p"], ["n"]),
        eval_data_fn=lambda: (["p"], ["n"]),
        target_tokens=["I"],
        detection_phrases=["I cannot"],
    )
    assert concept2.search_config["layer_cutoff_frac"] == DEFAULT_SEARCH_CONFIG["layer_cutoff_frac"]

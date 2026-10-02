"""Config plumbing: --json defaults merge, pre-split data shape detection, and
ConceptDefinition's search_config default.

Pure config/data-shape tests — no model is loaded.
"""
import copy
import json

import pytest

from run_experiment import build_default_config, apply_config_defaults, parse_args
from framework import normalize_train_data_result
from concept import ConceptDefinition, DEFAULT_SEARCH_CONFIG


# --- --json defaults merge ---

def test_minimal_json_config_gets_all_defaults():
    """The docs' own minimal example, through parse_args(), gets every argparse default."""
    defaults = build_default_config()
    config = parse_args(["--json", json.dumps({"model_name": "x", "mode": "search"})])
    assert config["model_name"] == "x"
    assert config["mode"] == "search"
    assert config["torch_dtype"] == "auto"
    assert config["force_cpu"] is False
    assert config["filter_prompts"] is True
    assert config["concept"] == "refusal"
    assert config["induce_mode"] == "single_layer"
    assert config["alpaca_max_prompts"] == 500
    for key in defaults:
        assert key in config, f"missing default key: {key}"


def test_unknown_json_key_raises():
    with pytest.raises(ValueError, match="Unknown config key"):
        apply_config_defaults({"model_name": "x", "mode": "search", "bogus_typo_key": 1})
    with pytest.raises(SystemExit):
        parse_args(["--json", json.dumps({"model_name": "x", "mode": "search", "bogus_typo_key": 1})])


def test_json_override_wins_over_default():
    config = apply_config_defaults({"model_name": "x", "mode": "search", "filter_prompts": False})
    assert config["filter_prompts"] is False

    config2 = apply_config_defaults({"model_name": "x", "mode": "search", "concept": "sycophancy"})
    assert config2["concept"] == "sycophancy"


# --- pre-split data shape detection ---

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


@pytest.mark.parametrize("result,match", [
    ({"positive": ["p1"], "negative": ["n1"]}, "unrecognized shape"),
    (12345, "my_concept"),                                    # the error names the concept
    (((["tp1"], ["tn1"]), (1, 2)), None),                     # nested non-strings aren't pre-split
])
def test_normalize_malformed_shape_raises(result, match):
    with pytest.raises(ValueError, match=match):
        normalize_train_data_result(result, "my_concept")


# --- ConceptDefinition default search_config ---

def _concept(name):
    return ConceptDefinition(
        name=name,
        train_data_fn=lambda: (["p"], ["n"]),
        eval_data_fn=lambda: (["p"], ["n"]),
        target_tokens=["I"],
        detection_phrases=["I cannot"],
    )


def test_concept_definition_default_search_config_is_independent_copy():
    """Defaults to DEFAULT_SEARCH_CONFIG, as a fresh copy per concept (a shared
    mutable default would let one concept's override leak into every other)."""
    concept = _concept("test_concept")
    assert concept.search_config == DEFAULT_SEARCH_CONFIG
    original = copy.deepcopy(DEFAULT_SEARCH_CONFIG)
    concept.search_config["layer_cutoff_frac"] = 0.1234
    assert DEFAULT_SEARCH_CONFIG == original
    assert _concept("test_concept_2").search_config == original

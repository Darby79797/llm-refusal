"""scripts/limit_factorial.py (the capability × knowledge factorial) and its prompt set. No model is loaded."""
import importlib.util
import os
import re

import prompts


def _script():
    spec = importlib.util.spec_from_file_location(
        "limit_factorial", os.path.join(os.path.dirname(__file__), "..", "scripts", "limit_factorial.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_cells_are_row_aligned_and_cross_the_factors():
    d = prompts.create_limit_factorial_data()
    assert list(d) == ["neither", "cant", "dontknow", "both"] and all(len(v) == 32 for v in d.values())
    for (real, invented, physical, text), *row in zip(prompts._LIMIT_FACTORIAL_ROWS, *d.values()):
        neither, cant, dontknow, both = row
        assert real.lower() in neither.lower() and real.lower() in cant.lower()
        assert invented.lower() in dontknow.lower() and invented.lower() in both.lower()
        assert neither.lower() == text.format(real).lower() and both.lower() == physical.format(invented).lower()
    assert len(set(sum(d.values(), []))) == 128


def test_invented_names_are_not_in_the_other_prompt_sets():
    src = open(prompts.__file__).read().split("_LIMIT_FACTORIAL_ROWS = [")[0]
    common = {"Tower", "Museum", "London", "Park", "University", "Motors", "Bridge", "Library", "Opera", "House",
              "Festival", "Lantern", "Stadium", "Statue", "Cathedral", "Saint", "River", "Mount", "Gorge", "Montana",
              "Brasserie", "Lyon", "Society", "Copper", "Keeper", "Orchard", "Lanterns", "Helen", "Peter", "Agnes",
              "Zoo", "Salt", "The"}
    for _, invented, _, _ in prompts._LIMIT_FACTORIAL_ROWS:
        for w in re.findall(r"[A-Z][a-z]+(?:-[A-Z][a-z]+)?", invented):
            assert w in common or w not in src, (w, invented)


def test_factor_aurocs_separate_the_two_factors():
    lf = _script()
    cap = {"neither": [0.0, 0.1], "dontknow": [0.05, 0.15], "cant": [1.0, 1.1], "both": [1.05, 1.2]}
    got = lf.factor_aurocs(cap)
    assert got["capability"] == 1.0 and got["capability|real"] == 1.0 and got["capability|invented"] == 1.0
    assert 0.25 <= got["knowledge"] <= 0.75
    know = {"neither": [0.0], "cant": [0.1], "dontknow": [2.0], "both": [2.1]}
    assert lf.factor_aurocs(know)["knowledge|physical"] == 1.0 and lf.factor_aurocs(know)["capability"] == 0.75


def test_flags_parse_rates_and_table():
    lf = _script()
    f = lf.parse_flags('{"cant": true, "dontknow": false, "declines": false, "pretends": false, "asserts": true}')
    assert f == {"cant": True, "dontknow": False, "declines": False, "pretends": False, "asserts": True}
    assert lf.parse_flags('{"cant": true}') is None and lf.parse_flags("nonsense") is None
    r = lf.rates([f, None, dict(f, cant=False)], {"inability": [True, False, False, False], "unknown": [False] * 4})
    assert r["cant"] == 50 and r["asserts"] == 100 and r["det_inability"] == 25 and r["det_unknown"] == 0
    out = {"generations": {"baseline": {"rates": {"both": r}}}}
    assert "baseline     both" in lf.table(out)

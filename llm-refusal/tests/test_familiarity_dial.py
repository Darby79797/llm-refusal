"""familiarity_dial.py: prompt sets, the flip AUROC and the summary table (no model)."""
import importlib.util
import os


def _mod():
    spec = importlib.util.spec_from_file_location(
        "familiarity_dial", os.path.join(os.path.dirname(__file__), "..", "scripts", "familiarity_dial.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_prompt_sets_are_equal_sized_and_disjoint():
    fd = _mod()
    d = fd.prompt_sets()
    assert list(d) == fd.SETS and len({len(v) for v in d.values()}) == 1
    assert not (set(d["invented"]) & set(d["obscure"])) and not (set(d["obscure"]) & set(d["famous"]))


def test_flipped_auroc_ignores_prompts_already_saying_dont_know():
    fd = _mod()
    proj = [5.0, 4.0, 1.0, 0.0, 9.0]
    base = [False, False, False, False, True]          # the last one said "don't know" already: left out
    cond = [True, True, False, False, True]
    assert fd.flipped_auroc(proj, base, cond) == 1.0    # the flipped ones are the more unfamiliar
    assert fd.flipped_auroc(proj, base, [False] * 5) is None


def test_summarise_prints_rates_and_flip_auroc():
    fd = _mod()
    g = lambda dk: {"dontknow": {"invented": [True, True], "obscure": dk, "famous": [False, False]},
                    "asserts": {s: [False, False] for s in fd.SETS}}
    out = {"projections": {"invented": [3, 3], "obscure": [2.0, 1.0], "famous": [0.0, -1.0]},
           "generations": {"baseline": g([False, False]), "add50": g([True, False])}}
    lines = fd.summarise(out)
    assert len(lines) == 3 and "50%" in lines[2] and lines[2].rstrip().endswith("1.00")

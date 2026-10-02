"""Index of saved run artifacts, so tooling never re-parses file names or logs itself.

Discovers, under one or more roots (default `results/`):
- evaluate runs: `<model>-<concept>-evaluate-<tag>-generations.json`, joined with the
  matching `...-evaluate-<tag>.log` (batch size/dtype header, log-odds, filter counts)
- searches: `<model>-<concept>-search-scores.csv` + `...-search.log` (selection tier)
- directions: `<model>-<concept>-direction.json`
- CAA sweeps: `caa/<model>-ab.json`; cross-concept: `<model>-cross_concept-*.json`

A leading `filtered-` / `rerun-` / `prefix-` etc. on a file name is kept as the run's
`variant`, so archived copies sit beside current ones instead of overwriting them.
Nothing here loads a model.
"""
import glob
import json
import math
import os
import re
from dataclasses import dataclass, field
from functools import cached_property
from typing import Dict, List, Optional, Tuple

CONCEPTS = ["refusal_arditi_exact", "refusal_arditi", "refusal", "sycophancy_neutral", "sycophancy",
            "hedging_v2", "hedging", "empathy"]
_CONCEPT_RE = "|".join(CONCEPTS)
_VARIANT_RE = r"(?:(?P<variant>[a-z][a-z0-9]*)-(?=[A-Z]))?"
EVAL_RE = re.compile(rf"^{_VARIANT_RE}(?P<model>.+?)-(?P<concept>{_CONCEPT_RE})-evaluate-(?P<tag>.+)-generations\.json$")
SEARCH_RE = re.compile(rf"^{_VARIANT_RE}(?P<model>.+?)-(?P<concept>{_CONCEPT_RE})-search-scores\.csv$")
DIRECTION_RE = re.compile(rf"^(?P<model>.+?)-(?P<concept>{_CONCEPT_RE})-direction\.json$")
TAG_RE = re.compile(r"^L(?P<layer>\d+)-P(?P<pos>-\d+)(?:-T(?P<tokens>\d+))?$")

# condition -> which prompt set it runs on (mirrors evaluation.CONDITIONS, without importing torch)
SIDE = {
    "baseline": "positive", "baseline_negative": "negative",
    "global_ablation": "positive", "layer_specific_ablation": "positive",
    "layer_specific_subtraction": "positive",
    "layer_specific_addition": "negative", "global_addition": "negative",
}
BASELINE_FOR = {"positive": "baseline", "negative": "baseline_negative"}
CONDITION_ORDER = ["baseline", "global_ablation", "layer_specific_ablation", "layer_specific_subtraction",
                   "baseline_negative", "layer_specific_addition", "global_addition"]
SHORT = {"baseline": "base", "global_ablation": "abl", "layer_specific_ablation": "L-abl",
         "layer_specific_subtraction": "L-sub", "baseline_negative": "neg-base",
         "layer_specific_addition": "add", "global_addition": "G-add"}

Z = 1.959964  # 95%


def detector(concept_name: str):
    """The concept's current phrase/heuristic detector (same logic as
    BigEvaluator._check_for_detection without an API judge), for relabeling saved
    responses after a detector fix. concept.py is torch-free, so this stays cheap."""
    from concept import get_concept
    concept = get_concept(concept_name)

    def detect(text: str) -> bool:
        text = text.replace("\u2019", "'").replace("\u2018", "'")
        if concept.detection_fn is not None:
            return bool(concept.detection_fn(text))
        return any(p.lower() in text.lower() for p in concept.detection_phrases)
    return detect


def wilson(k: int, n: int) -> Tuple[float, float]:
    """95% Wilson interval: rates here sit near 0/100%, where the normal interval collapses."""
    if n == 0:
        return float("nan"), float("nan")
    p = k / n
    denom = 1 + Z ** 2 / n
    centre = (p + Z ** 2 / (2 * n)) / denom
    half = Z * math.sqrt(p * (1 - p) / n + Z ** 2 / (4 * n ** 2)) / denom
    return max(0.0, centre - half), min(1.0, centre + half)


@dataclass
class Condition:
    name: str
    side: str
    n: int
    k: int
    degenerate: Optional[float] = None      # fraction flagged as repetition loops
    nll: Optional[float] = None             # mean response NLL under the clean model
    log_odds: Optional[float] = None        # from the evaluate log
    unsafe: Optional[float] = None          # LlamaGuard unsafe fraction, if scored

    @property
    def rate(self) -> float:
        return self.k / self.n if self.n else float("nan")

    @property
    def ci(self) -> Tuple[float, float]:
        return wilson(self.k, self.n)


def is_unsafe(label) -> bool:
    """LlamaGuard labels are its raw verdict, e.g. "unsafe\nS2" or "safe"."""
    return label is True or str(label).strip().lower().startswith("unsafe")


def phrase_artifact(base, cond, drop: float = 0.3) -> bool:
    """Detection rate fell by >= `drop` while the log-odds stayed positive: the phrasing
    changed but the decision did not (e.g. "I'm sorry, but I can't" -> "I do not provide").
    A real removal moves both (RESULTS.md: rate and log-odds must agree)."""
    if not base or not cond or base.log_odds is None or cond.log_odds is None:
        return False
    return base.rate - cond.rate >= drop and base.log_odds > 0 and cond.log_odds > 0


@dataclass
class EvalRun:
    root: str
    model: str
    concept: str
    tag: str
    gen_path: str
    variant: str = ""
    log_path: Optional[str] = None
    layer: Optional[int] = None
    pos: Optional[int] = None
    max_new_tokens: int = 64
    header: Dict[str, str] = field(default_factory=dict)     # gen_batch_size, dtype, ...
    filter: Optional[Dict[str, int]] = None                   # pos_in/pos_out/neg_in/neg_out, or {} if disabled
    oom_split_warning: bool = False
    conditions: Dict[str, Condition] = field(default_factory=dict)

    @property
    def id(self) -> str:
        v = f"{self.variant}:" if self.variant else ""
        return f"{v}{self.model}/{self.concept}/{self.tag}"

    @cached_property
    def generations(self) -> Dict[str, List[Dict]]:
        with open(self.gen_path) as f:
            return json.load(f)

    def effect(self, condition: str) -> Optional[str]:
        """'up'/'down' if the condition's CI clears its baseline's CI, else 'none'."""
        c = self.conditions.get(condition)
        b = self.conditions.get(BASELINE_FOR.get(c.side, "")) if c else None
        if not c or not b:
            return None
        (lo, hi), (blo, bhi) = c.ci, b.ci
        return "up" if lo > bhi else "down" if hi < blo else "none"

    def flags(self) -> List[str]:
        """Things a reader should know before trusting the numbers."""
        out = []
        for c in self.conditions.values():
            if c.degenerate and c.degenerate > 0.05:
                out.append(f"degenerate {SHORT.get(c.name, c.name)} {c.degenerate:.0%}")
        base = self.conditions.get("baseline")
        if base and base.n and base.rate < 0.2 and any(n in self.conditions for n in ("global_ablation", "layer_specific_ablation")):
            out.append(f"behavior rare at baseline ({base.rate:.0%}): ablation uninformative")
        if base and base.n < 30:
            out.append(f"small eval set (n={base.n})")
        if self.filter:
            kept = self.filter["pos_out"] / max(1, self.filter["pos_in"])
            if kept < 0.3:
                out.append(f"filter kept {self.filter['pos_out']}/{self.filter['pos_in']} positives")
        if self.oom_split_warning:
            out.append("OOM batch splits")
        for name in ("global_ablation", "layer_specific_ablation"):
            if phrase_artifact(self.conditions.get("baseline"), self.conditions.get(name)):
                out.append(f"{SHORT[name]}: rate fell but log-odds stayed > 0 (rewording, not removal?)")
        if "layer_specific_addition" in self.conditions and self.effect("layer_specific_addition") == "none":
            out.append("no induction")
        if "global_ablation" in self.conditions and self.effect("global_ablation") == "none":
            out.append("no ablation effect")
        return out


@dataclass
class SearchRun:
    root: str
    model: str
    concept: str
    csv_path: str
    variant: str = ""
    log_path: Optional[str] = None
    selected: Optional[Tuple[int, int]] = None
    tier: Optional[str] = None          # "strict", a relaxed-tier label, or "fallback"

    @property
    def id(self) -> str:
        v = f"{self.variant}:" if self.variant else ""
        return f"{v}{self.model}/{self.concept}/search"

    @cached_property
    def rows(self) -> List[Dict[str, float]]:
        import csv
        out = []
        with open(self.csv_path) as f:
            for r in csv.DictReader(f):
                out.append({k: (float(v) if v not in ("", None) else float("nan")) for k, v in r.items()})
        return out


# ── Log parsing ──────────────────────────────────────────────────
_HEADER_RE = re.compile(r"gen_batch_size: (?P<bs>\S+), max_new_tokens: (?P<t>\d+), dtype: (?P<dtype>\S+)")
_FILTER_RE = re.compile(r"Filtering(?: complete)?:.*?positive (\d+)→(\d+).*?negative (\d+)→(\d+)")
# Evaluation-config line: the only header in pre-2026-09 logs, which lack the report header
_CONFIG_RE = re.compile(r"Evaluation config: gen_batch_size=(?P<bs>\d+).*?dtype=(?P<dtype>\S+)")


def _parse_eval_log(run: EvalRun, text: str) -> None:
    report = text[text.rfind("COMPREHENSIVE EVALUATION REPORT"):] if "COMPREHENSIVE EVALUATION REPORT" in text else ""
    m = _HEADER_RE.search(report)
    if m:
        run.header = {"gen_batch_size": m["bs"], "max_new_tokens": m["t"], "dtype": m["dtype"].replace("torch.", "")}
    else:
        m = _CONFIG_RE.search(text)
        if m:
            run.header = {"gen_batch_size": m["bs"], "dtype": m["dtype"].replace("torch.", "")}
    run.oom_split_warning = "ran out of memory and were split" in report
    for section in re.split(r"\n--- ", report)[1:]:
        name = section.split(" ---")[0].strip().lower().replace(" ", "_")
        lo = re.search(r"log_odds_metric: (-?[\d.]+|nan)", section)
        if name in run.conditions and lo:
            run.conditions[name].log_odds = float(lo[1])
        lg = re.search(r"llamaguard_unsafe_rate: ([\d.]+)", section)
        if name in run.conditions and lg and run.conditions[name].unsafe is None:
            run.conditions[name].unsafe = float(lg[1])     # older runs saved no per-response labels
    filters = _FILTER_RE.findall(text)
    if filters:
        a, b, c, d = map(int, filters[-1])
        run.filter = {"pos_in": a, "pos_out": b, "neg_in": c, "neg_out": d}
    elif "disables behavioral prompt filtering" in text or "--no-filter-prompts" in text:
        run.filter = {}


def _parse_search_log(run: SearchRun, text: str) -> None:
    m = re.search(r"Strictly Selected Direction.*?\n.*?Layer: (\d+), Position: (-\d+)", text, re.S)
    if m:
        run.selected, run.tier = (int(m[1]), int(m[2])), "strict"
        return
    m = re.search(r"Relaxed selection \((.*?)\).*?Layer (\d+), Pos (-\d+)", text)
    if m:
        run.selected, run.tier = (int(m[2]), int(m[3])), m[1]
        return
    m = re.search(r"Falling back to best induce.*?Layer (\d+), Pos (-\d+)", text)
    if m:
        run.selected, run.tier = (int(m[1]), int(m[2])), "fallback"


def _sibling(path: str, variant: str, name: str) -> Optional[str]:
    """Log `name` next to `path`, honoring the variant prefix, else in results/."""
    d = os.path.dirname(path)
    for cand in ([os.path.join(d, f"{variant}-{name}")] if variant else []) + [os.path.join(d, name)]:
        if os.path.exists(cand):
            return cand
    return None


# ── Index ────────────────────────────────────────────────────────
@dataclass
class Index:
    evals: List[EvalRun]
    searches: List[SearchRun]
    directions: Dict[Tuple[str, str], Dict]
    caa: Dict[str, str]                # model -> caa/<model>-ab.json path
    cross: Dict[str, List[str]]        # model -> cross_concept json paths

    def find_evals(self, model: str = "", concept: str = "", tag: str = "", variant: Optional[str] = None) -> List[EvalRun]:
        """Substring match on model and tag; an exact tag match wins over a substring one."""
        hits = [r for r in self.evals
                if model.lower() in r.model.lower() and (not concept or r.concept == concept)
                and tag in r.tag and (variant is None or r.variant == variant)]
        exact = [r for r in hits if r.tag == tag]
        return exact or hits

    def find_search(self, model: str, concept: str, variant: str = "") -> Optional[SearchRun]:
        hits = [s for s in self.searches if model.lower() in s.model.lower() and s.concept == concept
                and s.variant == variant]
        return hits[0] if hits else None


def load_index(roots: Optional[List[str]] = None) -> Index:
    roots = roots or ["results"]
    evals, searches, directions, caa, cross = [], [], {}, {}, {}
    for root in roots:
        for path in sorted(glob.glob(os.path.join(root, "*-generations.json"))):
            m = EVAL_RE.match(os.path.basename(path))
            if not m:
                continue
            variant = m["variant"] or ""
            run = EvalRun(root=root, model=m["model"], concept=m["concept"], tag=m["tag"],
                          gen_path=path, variant=variant)
            t = TAG_RE.match(run.tag)
            if t:
                run.layer, run.pos = int(t["layer"]), int(t["pos"])
                run.max_new_tokens = int(t["tokens"] or 64)
            gens = run.generations
            for name in sorted(gens, key=lambda c: CONDITION_ORDER.index(c) if c in CONDITION_ORDER else 99):
                entries = gens[name]
                deg = [e["degenerate"] for e in entries if "degenerate" in e]
                nll = [e["nll"] for e in entries if isinstance(e.get("nll"), (int, float)) and not math.isnan(e["nll"])]
                lg = [e["llamaguard"] for e in entries if "llamaguard" in e and e["llamaguard"] is not None]
                run.conditions[name] = Condition(
                    name=name, side=SIDE.get(name, "?"), n=len(entries), k=sum(bool(e["detected"]) for e in entries),
                    degenerate=sum(deg) / len(deg) if deg else None,
                    nll=sum(nll) / len(nll) if nll else None,
                    unsafe=sum(1 for x in lg if is_unsafe(x)) / len(lg) if lg else None)
            del run.__dict__["generations"]   # loaded again lazily when a viewer asks
            # Logs carry the -T<n> suffix since 2026-09-30; older ones don't.
            run.log_path = (_sibling(path, variant, f"{run.model}-{run.concept}-evaluate-{run.tag}.log")
                            or _sibling(path, variant, f"{run.model}-{run.concept}-evaluate-{run.tag.split('-T')[0]}.log"))
            if run.log_path:
                with open(run.log_path, errors="replace") as f:
                    _parse_eval_log(run, f.read())
            evals.append(run)
        for path in sorted(glob.glob(os.path.join(root, "*-search-scores.csv"))):
            m = SEARCH_RE.match(os.path.basename(path))
            if not m:
                continue
            variant = m["variant"] or ""
            s = SearchRun(root=root, model=m["model"], concept=m["concept"], csv_path=path, variant=variant)
            s.log_path = _sibling(path, variant, f"{s.model}-{s.concept}-search.log")
            if s.log_path:
                with open(s.log_path, errors="replace") as f:
                    _parse_search_log(s, f.read())
            searches.append(s)
        for path in glob.glob(os.path.join(root, "*-direction.json")):
            m = DIRECTION_RE.match(os.path.basename(path))
            if m:
                with open(path) as f:
                    directions[(m["model"], m["concept"])] = {**json.load(f), "path": path[:-5]}
        for path in glob.glob(os.path.join(root, "caa", "*-ab*.json")):
            stem = os.path.basename(path)[:-len(".json")]          # <model>-ab or <model>-ab-<tag>
            if not stem.startswith("partial"):
                caa[stem.replace("-ab", "", 1)] = path
        for path in glob.glob(os.path.join(root, "*-cross_concept-*.json")):
            cross.setdefault(os.path.basename(path).split("-cross_concept-")[0], []).append(path)
    return Index(evals, searches, directions, caa, cross)

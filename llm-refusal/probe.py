"""Shared helpers for the analysis scripts in scripts/.

Setup: `load_run` (framework + model handles + the saved direction r / r̂ + short
name), `inhibitor_direction` (û and û⊥ from saved rank1 adapters), `save_json`.
Forward-pass probes: `residuals_at` (per-prompt residuals at a template position, every
layer), `token_projections` (attention-sink-masked per-token projection at one block),
`behaviour` (detection rate / log-odds under a context; the only thing here that
decodes). Small maths: `auroc`, `cos`, `unit`, `mean`, `split_by`.
"""
import contextlib
import json
import os
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import torch as t

from datatypes import DirectionVector
from finetune import load_adapters
from formatting import last_real_token_indices


# --- setup -------------------------------------------------------------------------

@dataclass
class Run:
    """A loaded model and its saved direction, as every analysis script starts."""
    fw: Any                         # DirectionTestFramework
    model: Any
    fmt: Any                        # fw.prompt_formatter
    ev: Any                         # fw.evaluator
    applier: Any                    # fw.intervention_applier
    blocks: Any                     # the transformer layers
    r: Optional[DirectionVector]
    r_hat: Optional[t.Tensor]       # r's unit vector, fp32 on the CPU
    short: str                      # model name without the org, e.g. Qwen2.5-0.5B-Instruct
    concept: str

    @property
    def coords(self) -> Dict[str, int]:
        """r's coordinates as the result JSONs record them."""
        return {"layer": self.r.layer, "position_index": self.r.position_index}

    def path(self, area: str, *parts: str) -> str:
        """results/<area>/<short>[-<part>...].json (empty parts dropped)."""
        return os.path.join("results", area, "-".join([self.short, *(p for p in parts if p)]) + ".json")

    def ablate(self, v) -> Callable:
        """Context factory: `v` (tensor or DirectionVector) hook-ablated at every layer."""
        return lambda: self.applier.intervened(v, "ablate", layers=None)

    def add(self, v: t.Tensor) -> Callable:
        """Context factory: `v` (unnormalised) added at r's layer and position."""
        dv = DirectionVector(vector=v, layer=self.r.layer, position_index=self.r.position_index, score=0)
        return lambda: self.applier.intervened(dv, "add", layers=[self.r.layer])


def load_run(model_name: str, concept: str = "refusal", direction: Union[bool, str] = True, **framework_kwargs) -> Run:
    """Load the framework for `model_name`/`concept` and the saved direction.

    direction: True loads results/<short>-<concept>-direction, a string loads that stem,
    False loads none (r and r_hat are then None). framework_kwargs go to
    DirectionTestFramework (torch_dtype, llamaguard_api_base, ...)."""
    from framework import DirectionTestFramework  # heavy (transformers); only scripts need it
    short = model_name.split("/")[-1]
    fw = DirectionTestFramework(model_name=model_name, concept=concept, **framework_kwargs)
    r = r_hat = None
    if direction:
        r = DirectionVector.load(direction if isinstance(direction, str) else f"results/{short}-{concept}-direction")
        r_hat = r.unit.float().cpu()
    applier = fw.intervention_applier
    return Run(fw=fw, model=fw.model, fmt=fw.prompt_formatter, ev=fw.evaluator, applier=applier,
               blocks=applier.transformer_layers, r=r, r_hat=r_hat, short=short, concept=concept)


def inhibitor_direction(stems: Sequence[str], r_hat: t.Tensor) -> Tuple[t.Tensor, t.Tensor]:
    """(û, û⊥) from saved rank1 adapter runs: û is the unit mean of the seeds' unit U
    columns, sign-aligned to the first (u vᵀ = (-u)(-v)ᵀ), and û⊥ its r̂-free part, unit
    (the direction the inhibitor writes). One stem gives that adapter's own u."""
    us = [load_adapters(stem)[1][0]["U"][:, 0].float() for stem in stems]
    us = [u * t.sign(u @ us[0]) for u in us]
    u_hat = unit(t.stack([unit(u) for u in us]).mean(0))
    return u_hat, unit(u_hat - (u_hat @ r_hat) * r_hat)


def save_json(path: str, obj) -> str:
    """Write `obj` to `path` (indent=1), creating the directory; returns `path`. Atomic (temp file + rename),
    so a crash mid-write leaves the previous version for resumable scripts to pick up."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=1)
    os.replace(tmp, path)
    return path


# --- forward-pass probes -----------------------------------------------------------

def _forward(model, enc) -> None:
    dev = model.device
    model(input_ids=enc["input_ids"].to(dev), attention_mask=enc["attention_mask"].to(dev),
          position_ids=enc["position_ids"].to(dev))


@t.no_grad()
def residuals_at(model, formatter, blocks, prompts: List[str], pos: int, batch_size: int = 16) -> t.Tensor:
    """[n_prompts, n_layers, d_model] fp32 (CPU): the residual stream entering each block
    (the point directions are extracted at) at template position `pos` (-1 = the last
    real token), under whatever hooks/edits are active."""
    out = []
    for i in range(0, len(prompts), batch_size):
        enc = formatter.format_batch(prompts[i:i + batch_size])
        idx = last_real_token_indices(enc["attention_mask"]) + 1 + pos
        rows = t.arange(len(idx))
        captured: List[Optional[t.Tensor]] = [None] * len(blocks)

        def make(l):
            def hook(module, args):
                captured[l] = args[0][rows.to(args[0].device), idx.to(args[0].device)].float().cpu()
            return hook
        handles = [b.register_forward_pre_hook(make(l)) for l, b in enumerate(blocks)]
        try:
            _forward(model, enc)
        finally:
            for h in handles:
                h.remove()
        out.append(t.stack(captured, dim=1))
    return t.cat(out)


@t.no_grad()
def token_projections(model, formatter, block, direction: t.Tensor, prompts: List[str],
                      batch_size: int = 16) -> Tuple[List[float], List[float]]:
    """Per prompt: the projection onto `direction` at every real token of the formatted
    prompt, read at `block`'s input. Returns (max over tokens, value at the last token)
    lists. Attention-sink tokens (BOS / the first template token) have residual norms
    10-100x the rest and dominate any projection, so tokens whose norm exceeds 4x the
    row's median are dropped from the max."""
    mx, last = [], []
    for i in range(0, len(prompts), batch_size):
        enc = formatter.format_batch(prompts[i:i + batch_size])
        mask = enc["attention_mask"]
        captured = {}

        def grab(m, a):
            x = a[0].float()
            captured["p"] = (x @ direction.to(x.device)).cpu()
            captured["n"] = x.norm(dim=-1).cpu()
        h = block.register_forward_pre_hook(grab)
        try:
            _forward(model, enc)
        finally:
            h.remove()
        norms = captured["n"].masked_fill(mask == 0, float("nan"))
        med = t.nanmedian(norms, dim=1).values[:, None]
        keep = (mask == 1) & (captured["n"] <= 4 * med)
        proj = captured["p"].masked_fill(~keep, float("-inf"))
        mx += proj.max(dim=1).values.tolist()
        idx = last_real_token_indices(mask)
        last += captured["p"][t.arange(len(idx)), idx].tolist()   # unmasked: the boundary is never dropped
    return mx, last


def behaviour(fw, prompts: List[str], ctx: Callable = contextlib.nullcontext, max_new_tokens: int = 64,
              texts_out: Optional[List[str]] = None, labels_out: Optional[List[bool]] = None) -> Dict[str, float]:
    """Detection rate and mean log-odds on `prompts` under `ctx()` (a context-manager
    factory), with the texts/labels optionally returned through the *_out lists."""
    ev = fw.evaluator
    with ctx():
        texts = ev.generate_responses(prompts, max_new_tokens=max_new_tokens)
        lo = ev._log_odds_metric(prompts)
    labels = [bool(ev._check_for_detection(x)) for x in texts]
    if texts_out is not None:
        texts_out.extend(texts)
    if labels_out is not None:
        labels_out.extend(labels)
    return {"rate": sum(labels) / len(labels) if labels else float("nan"), "log_odds": lo, "n": len(labels)}


# --- small maths -------------------------------------------------------------------

def auroc(pos, neg) -> float:
    """P(score of a random positive > a random negative); ties count half."""
    pos, neg = t.as_tensor(pos, dtype=t.float32), t.as_tensor(neg, dtype=t.float32)
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    diff = pos[:, None] - neg[None, :]
    return float((diff > 0).float().mean() + 0.5 * (diff == 0).float().mean())


def cos(a: t.Tensor, b: t.Tensor) -> float:
    return float(t.nn.functional.cosine_similarity(a.float().cpu().flatten(), b.float().cpu().flatten(), dim=0))


def unit(v: t.Tensor) -> t.Tensor:
    v = v.float().cpu()
    return v / v.norm()


def mean(xs) -> Optional[float]:
    """sum/len, or None for an empty list."""
    xs = list(xs)
    return sum(xs) / len(xs) if xs else None


def split_by(xs, labels) -> Tuple[list, list]:
    """(xs where the label is true, xs where it is false)."""
    return [x for x, l in zip(xs, labels) if l], [x for x, l in zip(xs, labels) if not l]

"""Small forward-pass probes shared by the analysis scripts: per-prompt residuals at a
template position for every layer, behaviour (detection rate / log-odds) under a
context, AUROC. Nothing here decodes except `behaviour`.
"""
import contextlib
from typing import Callable, Dict, List, Optional

import torch as t

from formatting import last_real_token_indices


@t.no_grad()
def residuals_at(model, formatter, blocks, prompts: List[str], pos: int, batch_size: int = 16) -> t.Tensor:
    """[n_prompts, n_layers, d_model] fp32 (CPU): the residual stream entering each block
    (the point directions are extracted at) at template position `pos` (-1 = the last
    real token), under whatever hooks/edits are active."""
    dev = model.device
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
            model(input_ids=enc["input_ids"].to(dev), attention_mask=enc["attention_mask"].to(dev),
                  position_ids=enc["position_ids"].to(dev))
        finally:
            for h in handles:
                h.remove()
        out.append(t.stack(captured, dim=1))
    return t.cat(out)


def auroc(pos, neg) -> float:
    """P(score of a random positive > a random negative); ties count half."""
    pos, neg = t.as_tensor(pos, dtype=t.float32), t.as_tensor(neg, dtype=t.float32)
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    diff = pos[:, None] - neg[None, :]
    return float((diff > 0).float().mean() + 0.5 * (diff == 0).float().mean())


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


def cos(a: t.Tensor, b: t.Tensor) -> float:
    return float(t.nn.functional.cosine_similarity(a.float().cpu().flatten(), b.float().cpu().flatten(), dim=0))


def unit(v: t.Tensor) -> t.Tensor:
    v = v.float().cpu()
    return v / v.norm()

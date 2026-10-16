"""Direction specs: one string grammar for "which direction(s)" shared by the analysis scripts.

  STEM                 a saved DirectionVector (results/...-direction)
  rhat                 the run's refusal direction r
  random@STEM          a norm-matched random direction at that stem's layer/position (ablation ignores the norm)
  span:A+B+...         those directions' whole span, as one [k, d] stack (removed in one projection)
  randspan:K@STEM      K random directions' span
  pcs:K@A+B+...        the top K principal axes of those directions (unit-normed, uncentred: the first is close to
                       their mean)
  pc:K1/K2/...@A+B+... only those principal axes, 1-based
  tilt:C:N@SPEC        draw N of a random unit direction ⊥ r̂ with cosine exactly C to SPEC's first axis
  A&B&...              several specs' axes together
  perp:SPEC            SPEC with r̂ projected out

`load_spec` returns a DirectionVector for single saved/derived directions and a [k, d] fp32 CPU tensor for stacks;
`rows` turns either into [k, d]. Random specs take their seed from `seed_of(spec)`.
"""
import contextlib
import json
from typing import Callable, List, Optional, Tuple, Union

import torch as t

from datatypes import DirectionVector

Spec = Union[DirectionVector, t.Tensor]


def unit(v: t.Tensor) -> t.Tensor:
    return v / v.norm()


def rows(d: Spec) -> t.Tensor:
    """[k, d] fp32 CPU rows of a loaded spec."""
    return d.float() if isinstance(d, t.Tensor) else d.vector.float().cpu()[None]


def load_spec(spec: str, r: DirectionVector, seed_of: Callable[[str], int] = lambda spec: 0) -> Spec:
    if "&" in spec:
        return t.cat([rows(load_spec(x, r, seed_of)) for x in spec.split("&")])
    if spec == "rhat":
        return r
    u = unit(r.vector.float().cpu())
    if spec.startswith("random@"):
        ref = DirectionVector.load(spec[7:])
        g = t.Generator().manual_seed(seed_of(spec))
        v = unit(t.randn(ref.vector.shape[-1], generator=g)) * ref.vector.float().norm()
        return DirectionVector(vector=v.to(ref.vector.dtype), layer=ref.layer, position_index=ref.position_index, score=0)
    if spec.startswith("span:"):
        return t.stack([rows(load_spec(x, r, seed_of))[0] for x in spec[5:].split("+")])
    if spec.startswith(("pcs:", "pc:")):
        ks, stems = spec.split(":", 1)[1].split("@", 1)
        Vh = principal_axes([load_spec(x, r, seed_of) for x in stems.split("+")])
        return Vh[:int(ks)] if spec.startswith("pcs:") else Vh[[int(k) - 1 for k in ks.split("/")]]
    if spec.startswith("tilt:"):
        cn, ref = spec[5:].split("@", 1)
        c = float(cn.split(":")[0])
        a0 = rows(load_spec(ref, r, seed_of))[0]
        a0 = unit(a0 - (a0 @ u) * u)
        g = t.randn(a0.shape[-1], generator=t.Generator().manual_seed(seed_of(spec)))
        g = g - (g @ u) * u
        g = unit(g - (g @ a0) * a0)
        return (c * a0 + (1 - c * c) ** 0.5 * g)[None]
    if spec.startswith("perp:"):
        d = load_spec(spec[5:], r, seed_of)
        if isinstance(d, t.Tensor):
            return d.float() - (d.float() @ u)[:, None] * u
        v = d.vector.float().cpu()
        return DirectionVector(vector=(v - (v @ u) * u).to(d.vector.dtype), layer=d.layer,
                               position_index=d.position_index, score=0)
    if spec.startswith("randspan:"):
        k, ref = spec[9:].split("@", 1)
        g = t.Generator().manual_seed(seed_of(spec))
        return t.randn(int(k), DirectionVector.load(ref).vector.shape[-1], generator=g)
    return DirectionVector.load(spec)


def principal_axes(dirs: List[Spec]) -> t.Tensor:
    """Right singular vectors (rows) of the unit-normed directions, uncentred."""
    V = t.stack([unit(rows(d)[0]) for d in dirs])
    return t.linalg.svd(V, full_matrices=False).Vh


def signed_pc_axes(pool: str, r: DirectionVector, ks: List[int]) -> List[Tuple[int, t.Tensor]]:
    """(k, unit axis) for each 1-based k, signed so the pool's directions (harmful-minus-harmless contrasts) project
    positively on average: the harmful side is positive, as for every saved direction."""
    dirs = [load_spec(x, r) for x in pool.split("+")]
    Vh = principal_axes(dirs)
    V = t.stack([unit(rows(d)[0]) for d in dirs])
    out = []
    for k in ks:
        a = Vh[k - 1]
        out.append((k, a if (V @ a).sum() >= 0 else -a))
    return out


def save_pc_axes(pool: str, r: DirectionVector, ks: List[int], layer: int, out_prefix: str) -> List[str]:
    """Write each signed axis as <out_prefix>-PC<k>-direction (unit, `layer`, position -1) and the pool to
    <out_prefix>-axes.json; returns the stems."""
    stems = []
    for k, axis in signed_pc_axes(pool, r, ks):
        stem = f"{out_prefix}-PC{k}-direction"
        DirectionVector(vector=axis, layer=layer, position_index=-1, score=0).save(stem)
        stems.append(stem)
    with open(f"{out_prefix}-axes.json", "w") as f:
        json.dump({"pool": pool.split("+"), "ks": ks, "layer": layer, "stems": stems}, f, indent=1)
    return stems


def as_edit_vector(d: Spec, dtype) -> t.Tensor:
    """[d] or [k, d] tensor in `dtype` for orthogonalize.orthogonalized / the intervention applier."""
    return (d.vector if isinstance(d, DirectionVector) else d).to(dtype)


def push_vector(axis: t.Tensor, gap: float, scale: float) -> t.Tensor:
    """scale x the harmful-minus-harmless `gap` along `axis` (made unit): at S = 1 a harmless prompt's projection moves
    by as much as the harmful prompts sit above it."""
    return unit(axis) * gap * scale


def ablated(applier, spec: Optional[str], r: DirectionVector):
    """Context: SPEC's axes hook-ablated at every layer through `applier` (an InterventionApplier), or nothing for
    None / ""."""
    if not spec:
        return contextlib.nullcontext()
    return applier.intervened(as_edit_vector(load_spec(spec, r), r.vector.dtype), "ablate")


def save_mean_direction(stems: List[str], out_stem: str, layer: Optional[int] = None) -> DirectionVector:
    """Write the mean of the stems' unit directions, rescaled to their mean norm, as an ordinary stem (at `layer`, default
    the first stem's, position -1): a "shared" direction that can be added at strength 1 like its members."""
    dirs = [DirectionVector.load(x) for x in stems]
    v = unit(t.stack([unit(d.vector.float().cpu()) for d in dirs]).mean(0))
    v = v * t.stack([d.vector.float().norm() for d in dirs]).mean()
    out = DirectionVector(vector=v.to(dirs[0].vector.dtype), layer=dirs[0].layer if layer is None else layer,
                          position_index=-1, score=0)
    out.save(out_stem)
    return out


def save_random_like(stem: str, out_stem: str, seed: int) -> DirectionVector:
    """Write a random direction with STEM's norm, layer and position as an ordinary stem: the push-size control for
    adding / subtracting STEM at strength 1."""
    ref = DirectionVector.load(stem)
    g = t.Generator().manual_seed(seed)
    v = unit(t.randn(ref.vector.shape[-1], generator=g)) * ref.vector.float().norm()
    out = DirectionVector(vector=v.to(ref.vector.dtype), layer=ref.layer, position_index=ref.position_index, score=0)
    out.save(out_stem)
    return out


def save_moved(stem: str, out_stem: str, layer: Optional[int] = None, like: Optional[str] = None) -> DirectionVector:
    """Write STEM's direction at another layer: LIKE's layer and norm (a learned direction of the same push size, where
    LIKE is pushed), or LAYER with STEM's own norm (the same push elsewhere)."""
    assert (layer is None) != (like is None), "give exactly one of layer / like"
    ref = DirectionVector.load(stem)
    v = ref.vector.float().cpu()
    if like is not None:
        to = DirectionVector.load(like)
        v, layer = unit(v) * to.vector.float().norm(), to.layer
    out = DirectionVector(vector=v.to(ref.vector.dtype), layer=layer, position_index=ref.position_index, score=0)
    out.save(out_stem)
    return out


def save_projected_out(stem: str, away: List[str], out_stem: str) -> DirectionVector:
    """Write STEM with the span of the AWAY stems projected out, rescaled to STEM's norm (so a push at strength 1 is the
    same size), at STEM's layer and position."""
    ref = DirectionVector.load(stem)
    v = ref.vector.float().cpu()
    Q, _ = t.linalg.qr(t.stack([unit(DirectionVector.load(x).vector.float().cpu()) for x in away]).T)
    p = v - Q @ (Q.T @ v)
    p = unit(p) * v.norm()
    out = DirectionVector(vector=p.to(ref.vector.dtype), layer=ref.layer, position_index=ref.position_index, score=0)
    out.save(out_stem)
    return out

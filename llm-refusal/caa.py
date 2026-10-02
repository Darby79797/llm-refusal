"""Contrastive Activation Addition (Panickssery et al. 2023, "Steering Llama 2 via
Contrastive Activation Addition"): the multiple-choice (A/B) half, plus comparison of
CAA vectors with this repo's prompt-contrast directions.

Faithful to the reference implementation (github.com/nrimsky/CAA, commit 5dabbbd):

- **Contrast.** Each pair is one question with two appended answers, "(A)" and "(B)".
  The vector at layer L is mean(act_matching - act_not_matching) at the answer-letter
  token (index -2, before ")"). The prompts are identical; only the answer differs.
  This repo's own directions instead contrast *different prompts* at template tokens.
- **Layer convention.** CAA reads and steers the residual stream at the *output* of
  decoder block L. This repo's directions live at the *input* of block L. So CAA layer
  L is the same residual-stream point as our layer L+1. Everything in this module uses
  CAA's convention; conversions are explicit (`caa_layer_to_ours`).
- **Normalization.** At each layer, every behavior's vector is rescaled to the mean
  norm across behaviors at that layer, so one multiplier means the same size of push
  for every behavior.
- **Steering.** multiplier * vector is added to block L's output at every position
  from the last token of the prompt template onward (CAA: from the "]" of "[/INST]").
- **Metric.** p(answer matching behavior) = p(match) / (p(match) + p(not match)) for
  the next token after "(", averaged over the 50 held-out test questions. On the A/B
  items this is CAA's p(letter) / (p(A) + p(B)). CAA's plotting code scores the 4
  survival-instinct test items labelled (C)/(E) as 0 (it only checks for "A" or "B"),
  which biases that behavior down; `p_match_caa` reproduces that, for comparison with
  the paper's figures.

Batches are right-padded; the letter and logit indices come from
`formatting.last_real_token_indices`, never hand-rolled.
"""
import json
import logging
import os
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import torch as t

from batching import map_batched
from formatting import ChatPromptFormatter, last_real_token_indices, pad_rows

logger = logging.getLogger(__name__)

BEHAVIORS = [
    "coordinate-other-ais",
    "corrigible-neutral-HHH",
    "hallucination",
    "myopic-reward",
    "survival-instinct",
    "sycophancy",
    "refusal",
]
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "caa")


def caa_layer_to_ours(layer: int) -> int:
    """CAA layer L (output of block L) is our layer L+1 (input of block L+1)."""
    return layer + 1


def load_ab(behavior: str, split: str) -> List[Dict[str, str]]:
    """split: 'generate' (vector construction) or 'test' (50 held-out A/B questions)."""
    name = "generate_dataset.json" if split == "generate" else "test_dataset_ab.json"
    with open(os.path.join(DATA_DIR, behavior, name)) as f:
        return json.load(f)


def _letter(answer: str) -> str:
    """"(A)" -> "A". Mostly A/B, but survival-instinct has multi-option items whose
    labels run to (H): 83 of 903 generate pairs and 4 of 50 test questions."""
    letter = answer.strip().strip("()").strip()
    if len(letter) != 1 or not "A" <= letter <= "H":
        raise ValueError(f"Unexpected answer label {answer!r}")
    return letter


@dataclass
class Encoded:
    ids: List[int]      # full sequence, ending "(X)"
    boundary: int       # index of the last prompt-template token (steering starts here)


class CAAEncoder:
    """Builds CAA token sequences through this repo's chat templates.

    The answer is joined the way CAA joins it (`f"{prompt} {answer}"`): with a space,
    unless the template already ends in one (our Llama-2 template, "[INST] {x} [/INST] ",
    so Llama-2 sequences are character-identical to CAA's). The space matters: without
    it, Llama-3 and Qwen tokenize "(A" as one token after the template's newline, and
    there is no answer-letter position to read. Every sequence is checked, not assumed.
    """

    def __init__(self, tokenizer, formatter: ChatPromptFormatter):
        self.tokenizer = tokenizer
        self.formatter = formatter
        self.letter_ids: Dict[str, int] = {}

    def _tokenize(self, text: str) -> List[int]:
        ids = self.tokenizer.encode(text, add_special_tokens=False)
        if self.formatter.prepend_bos:
            ids = [self.tokenizer.bos_token_id] + ids
        return ids

    def encode(self, question: str, letter: str) -> Encoded:
        prompt = self.formatter.format_text(question.strip())
        sep = "" if prompt.endswith(" ") else " "
        ids = self._tokenize(prompt + sep + f"({letter})")
        prefix = self._tokenize(prompt.rstrip(" "))
        if ids[:len(prefix)] != prefix:
            raise ValueError("Prompt template does not tokenize as a prefix of prompt+answer; "
                             f"cannot locate the steering boundary: {prompt[-40:]!r}")
        letter_id = ids[-2]
        if self.tokenizer.decode([letter_id]).strip() != letter:
            raise ValueError(f"Answer letter is not its own token: "
                             f"{self.tokenizer.convert_ids_to_tokens(ids[-4:])}")
        known = self.letter_ids.setdefault(letter, letter_id)
        if known != letter_id:
            raise ValueError(f"Letter {letter} tokenized inconsistently ({known} vs {letter_id})")
        return Encoded(ids=ids, boundary=len(prefix) - 1)


def _pad(seqs: Sequence[List[int]], pad_id: int, device) -> Tuple[t.Tensor, t.Tensor, t.Tensor]:
    """Right-padded input_ids, attention_mask, position_ids, on `device`."""
    enc = pad_rows(list(seqs), pad_id)
    return enc['input_ids'].to(device), enc['attention_mask'].to(device), enc['position_ids'].to(device)


def _output_hook(fn: Callable[[t.Tensor], Optional[t.Tensor]]):
    """Forward hook on a decoder block that sees (and may replace) its hidden-state output."""
    def hook(module, args, output):
        hidden = output[0] if isinstance(output, tuple) else output
        new = fn(hidden)
        if new is None:
            return None
        return (new,) + tuple(output[1:]) if isinstance(output, tuple) else new
    return hook


class CAA:
    """Vector construction and A/B steering sweeps for one loaded model.

    Both hot paths share work through a KV cache instead of re-running whole prompts:
    - A/B sweep: steering only touches positions from the prompt boundary on (the
      last ~2 tokens), so each question's prefix is run once and every condition
      (baseline, layer x multiplier, cross-applied directions) runs only the suffix
      against that cache. That is ~65 conditions per prefix pass instead of one.
    - Vectors: the matching and non-matching sequences are identical up to the answer
      letter, so the shared prefix is run once and each letter is one cached token.
    `compute_vectors_full` / `ab_probs_full` are the direct (uncached) computations,
    kept as the reference the cached paths are tested against.
    """

    def __init__(self, model, tokenizer, formatter: ChatPromptFormatter, layers_modules,
                 batch_size: int = 64, max_batch_tokens: Optional[int] = None, on_split=None):
        self.model = model
        self.tokenizer = tokenizer
        self.encoder = CAAEncoder(tokenizer, formatter)
        self.blocks = layers_modules
        self.batch_size = batch_size                  # max rows per batch
        self.max_batch_tokens = max_batch_tokens      # max padded prefix tokens per batch (KV-cache bound)
        self.on_split = on_split
        self.pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
        # Decoder stack without the LM head: vectors need no logits at all, and the A/B
        # metric needs them at one position per row, not the full (seq x vocab) tensor.
        self.body = getattr(model, model.base_model_prefix)
        self.head = model.get_output_embeddings()

    @property
    def device(self):
        return next(self.model.parameters()).device

    # ── batching and cached forwards ─────────────────────────────
    def _batches(self, lengths: List[int]) -> List[List[int]]:
        """Length-sorted batches (a batch costs its longest row), each at most
        batch_size rows and max_batch_tokens padded tokens."""
        order = sorted(range(len(lengths)), key=lambda i: lengths[i])
        out, cur = [], []
        for i in order:
            too_many = len(cur) + 1 > self.batch_size
            too_long = self.max_batch_tokens is not None and lengths[i] * (len(cur) + 1) > self.max_batch_tokens
            if cur and (too_many or too_long):
                out.append(cur)
                cur = []
            cur.append(i)
        if cur:
            out.append(cur)
        return out

    def _each_batch(self, lengths: List[int], fn: Callable[[List[int]], None]) -> None:
        """fn(indices) per batch; a batch that runs out of memory is split and retried."""
        for batch in self._batches(lengths):
            map_batched(lambda idx: (fn(idx), [None] * len(idx))[1], batch, len(batch), on_split=self.on_split)

    def _prefix(self, seqs: List[List[int]]):
        ids, mask, pos = _pad(seqs, self.pad_id, self.device)
        last_real_token_indices(mask)                 # asserts right padding
        out = self.body(input_ids=ids, attention_mask=mask, position_ids=pos, use_cache=True)
        return out.past_key_values, mask

    def _suffix(self, cache, prefix_mask: t.Tensor, suffix: List[List[int]]) -> t.Tensor:
        """Run equal-length suffixes after right-padded cached prefixes (padding slots are
        masked out, positions continue from each row's real length), then crop the cache
        back so the next condition starts from the same prefix."""
        ids = t.tensor(suffix, dtype=t.long, device=self.device)
        B, S = ids.shape
        width = prefix_mask.shape[1]
        mask = t.cat([prefix_mask, t.ones(B, S, dtype=prefix_mask.dtype, device=self.device)], dim=1)
        pos = prefix_mask.sum(1, keepdim=True) + t.arange(S, device=self.device)[None, :]
        try:
            return self.body(input_ids=ids, attention_mask=mask, position_ids=pos,
                             past_key_values=cache, use_cache=True).last_hidden_state
        finally:
            cache.crop(width)

    # ── Vector construction ──────────────────────────────────────
    def _pairs(self, items):
        prefixes, letters = [], []
        for item in items:
            q = item["question"]
            pos = self.encoder.encode(q, _letter(item["answer_matching_behavior"])).ids
            neg = self.encoder.encode(q, _letter(item["answer_not_matching_behavior"])).ids
            if pos[:-2] != neg[:-2]:
                raise ValueError(f"Answer pair does not share a prefix: {q[:80]!r}")
            prefixes.append(pos[:-2])                 # ends at "("
            letters.append((pos[-2], neg[-2]))
        return prefixes, letters

    @t.no_grad()
    def compute_vectors(self, items: List[Dict[str, str]], desc: str = "") -> t.Tensor:
        """(n_layers, d_model) float64: mean over pairs of act(matching) - act(not matching)
        at the answer letter, at the output of every block. The trailing ")" of "(X)" is
        not run: it comes after the letter, so it cannot affect the letter's activation."""
        prefixes, letters = self._pairs(items)
        n_layers = len(self.blocks)
        total = t.zeros(n_layers, self.model.config.hidden_size, dtype=t.float64)

        def run(idx):
            nonlocal total
            cache, pmask = self._prefix([prefixes[i] for i in idx])
            diff = 0
            for sign, which in ((1.0, 0), (-1.0, 1)):
                captured = [None] * n_layers

                def make(layer):
                    def grab(hidden):
                        captured[layer] = hidden[:, 0].detach().to("cpu").to(t.float64)  # MPS has no float64
                    return _output_hook(grab)
                handles = [b.register_forward_hook(make(i)) for i, b in enumerate(self.blocks)]
                try:
                    self._suffix(cache, pmask, [[letters[i][which]] for i in idx])
                finally:
                    for h in handles:
                        h.remove()
                diff = diff + sign * t.stack(captured)           # (n_layers, B, d)
            total += diff.sum(1)

        self._each_batch([len(p) for p in prefixes], run)
        logger.info(f"CAA vectors {desc}: {len(items)} pairs")
        return total / len(items)

    @t.no_grad()
    def compute_vectors_full(self, items: List[Dict[str, str]]) -> t.Tensor:
        """Reference: CAA's computation verbatim (full forward of prompt + "(X)", read at -2)."""
        n_layers = len(self.blocks)
        total = t.zeros(n_layers, self.model.config.hidden_size, dtype=t.float64)
        for item in items:
            for sign, key in ((1.0, "answer_matching_behavior"), (-1.0, "answer_not_matching_behavior")):
                ids = self.encoder.encode(item["question"], _letter(item[key])).ids
                captured = [None] * n_layers

                def make(layer):
                    def grab(hidden):
                        captured[layer] = hidden[0, -2].detach().to("cpu").to(t.float64)
                    return _output_hook(grab)
                handles = [b.register_forward_hook(make(i)) for i, b in enumerate(self.blocks)]
                try:
                    self.body(input_ids=t.tensor([ids], device=self.device), use_cache=False)
                finally:
                    for h in handles:
                        h.remove()
                total += sign * t.stack(captured)
        return total / len(items)

    # ── A/B evaluation ───────────────────────────────────────────
    def _ab_rows(self, items, probs_by_item):
        out = []
        for item, p in zip(items, probs_by_item):
            match = _letter(item["answer_matching_behavior"])
            other = _letter(item["answer_not_matching_behavior"])
            ab = p["A"] + p["B"]
            out.append({"p_a": p["A"], "p_b": p["B"], "p_matching_letter": p[match],
                        "p_match": p[match] / (p[match] + p[other]),
                        "p_match_caa": p[match] / ab if match in "AB" else 0.0,
                        "ab_mass": ab})
        return out

    def _ab_encode(self, items):
        encoded = []
        for item in items:
            enc = self.encoder.encode(item["question"], _letter(item["answer_matching_behavior"]))
            self.encoder.encode(item["question"], _letter(item["answer_not_matching_behavior"]))
            encoded.append(Encoded(ids=enc.ids[:-2], boundary=enc.boundary))  # ends at "("
        for letter in "AB":
            self.encoder.encode(items[0]["question"], letter)
        return encoded

    @t.no_grad()
    def ab_sweep(self, items: List[Dict[str, str]],
                 conditions: List[Tuple[Optional[int], Optional[t.Tensor], float]]) -> List[List[Dict]]:
        """For each condition (layer, vector, multiplier) — (None, None, 0) is unsteered —
        per item p(A), p(B), p(matching). multiplier*vector is added to block `layer`'s
        output at every position from the prompt boundary on, i.e. the whole suffix."""
        encoded = self._ab_encode(items)
        prefixes = [e.ids[:e.boundary] for e in encoded]
        suffixes = [e.ids[e.boundary:] for e in encoded]
        if len({len(x) for x in suffixes}) != 1:
            raise ValueError(f"Suffix lengths differ across items: {sorted({len(x) for x in suffixes})}")
        ids_of = dict(self.encoder.letter_ids)
        results = [[None] * len(items) for _ in conditions]

        def run(idx):
            cache, pmask = self._prefix([prefixes[i] for i in idx])
            for c, (layer, vector, multiplier) in enumerate(conditions):
                handle = None
                if vector is not None and multiplier != 0.0:
                    def add(hidden, v=vector, m=multiplier):
                        return hidden + (m * v).to(device=hidden.device, dtype=hidden.dtype)
                    handle = self.blocks[layer].register_forward_hook(_output_hook(add))
                try:
                    hidden = self._suffix(cache, pmask, [suffixes[i] for i in idx])
                finally:
                    if handle is not None:
                        handle.remove()
                probs = t.softmax(self.head(hidden[:, -1]).float(), dim=-1)
                for i, p in zip(idx, probs):
                    results[c][i] = {k: p[j].item() for k, j in ids_of.items()}

        self._each_batch([len(p) for p in prefixes], run)
        return [self._ab_rows(items, r) for r in results]

    def ab_probs(self, items: List[Dict[str, str]], layer: Optional[int] = None,
                 vector: Optional[t.Tensor] = None, multiplier: float = 0.0) -> List[Dict]:
        return self.ab_sweep(items, [(layer, vector, multiplier)])[0]

    @t.no_grad()
    def ab_probs_full(self, items: List[Dict[str, str]], layer: Optional[int] = None,
                      vector: Optional[t.Tensor] = None, multiplier: float = 0.0) -> List[Dict]:
        """Reference: one uncached forward per item, steering positions >= boundary."""
        encoded = self._ab_encode(items)
        rows = []
        for e in encoded:
            handle = None
            if vector is not None and multiplier != 0.0:
                def add(hidden, b=e.boundary):
                    h = hidden.clone()
                    h[:, b:] += (multiplier * vector).to(device=h.device, dtype=h.dtype)
                    return h
                handle = self.blocks[layer].register_forward_hook(_output_hook(add))
            try:
                hidden = self.body(input_ids=t.tensor([e.ids], device=self.device), use_cache=False).last_hidden_state
            finally:
                if handle is not None:
                    handle.remove()
            p = t.softmax(self.head(hidden[0, -1]).float(), dim=-1)
            rows.append({k: p[j].item() for k, j in self.encoder.letter_ids.items()})
        return self._ab_rows(items, rows)


def normalize_across_behaviors(vectors: Dict[str, t.Tensor]) -> Dict[str, t.Tensor]:
    """CAA's normalize_vectors.py: at each layer, rescale every behavior's vector to the
    mean norm across behaviors at that layer. vectors: behavior -> (n_layers, d)."""
    norms = t.stack([v.norm(dim=-1) for v in vectors.values()])   # (n_behaviors, n_layers)
    mean = norms.mean(0)
    return {b: v * (mean / v.norm(dim=-1))[:, None] for b, v in vectors.items()}


def mean_p(results: List[Dict], key: str = "p_match") -> float:
    return sum(r[key] for r in results) / len(results)


def run_ab_sweep(caa: CAA, behaviors: List[str], layers: List[int], multipliers: List[float],
                 out_path: str, extra_vectors: Optional[Dict[str, Tuple[int, t.Tensor]]] = None,
                 vectors: Optional[Dict[str, t.Tensor]] = None) -> Dict:
    """Vectors for every behavior, then p(matching) on each behavior's test set at every
    (layer, multiplier). `extra_vectors` (name -> (caa_layer, direction)) are applied to
    every behavior's test set at their own layer, for cross-application, rescaled to the
    normalized CAA norm at that layer so that a multiplier means the same size of push."""
    reused = vectors is not None
    raw = vectors if reused else {b: caa.compute_vectors(load_ab(b, "generate"), desc=b) for b in behaviors}
    normalized = normalize_across_behaviors(raw) if len(behaviors) > 1 else raw
    caa_norm = next(iter(normalized.values())).norm(dim=-1)   # equal across behaviors
    extra_vectors = {name: (layer, v.to(t.float64) / v.to(t.float64).norm() * caa_norm[layer])
                     for name, (layer, v) in (extra_vectors or {}).items()}
    if not reused:   # (a tagged rerun reuses the model's vectors; it never overwrites them)
        vec_path = out_path.replace(".json", "-vectors.pt")
        t.save({"raw": raw, "normalized": normalized}, vec_path)
        logger.info(f"Saved CAA vectors to {vec_path}")

    results = {"caa_norm_by_layer": caa_norm.tolist(), "behaviors": behaviors, "layers": layers, "multipliers": multipliers,
               "batch_size": caa.batch_size, "max_batch_tokens": caa.max_batch_tokens,
               "dtype": str(caa.model.dtype), "method": "kv-cached prefix", "reused_vectors": reused,
               "baseline": {}, "sweep": {}, "cross": {}}
    for b in behaviors:
        test = load_ab(b, "test")
        conds = [("baseline", None, None, 0.0)]
        conds += [(("sweep", layer, m), layer, normalized[b][layer], m) for layer in layers for m in multipliers]
        conds += [(("cross", name, m), layer, vec, m) for name, (layer, vec) in extra_vectors.items() for m in multipliers]
        res = dict(zip([c[0] for c in conds], caa.ab_sweep(test, [c[1:] for c in conds])))
        entry = lambda r: {"p_match": mean_p(r), "p_match_caa": mean_p(r, "p_match_caa"), "items": r}
        base = res["baseline"]
        results["baseline"][b] = entry(base)
        logger.info(f"[{b}] baseline p(match)={mean_p(base):.3f} "
                    f"(A/B mass {sum(r['ab_mass'] for r in base) / len(base):.3f})")
        results["sweep"][b] = {str(layer): {str(m): entry(res[("sweep", layer, m)]) for m in multipliers}
                               for layer in layers}
        for layer in layers:
            logger.info(f"[{b}] L{layer}: " + "  ".join(
                f"x{m:+g}={mean_p(res[('sweep', layer, m)]):.3f}" for m in multipliers))
        for name, (layer, _) in extra_vectors.items():
            results["cross"].setdefault(name, {})[b] = {"layer": layer, **{str(m): entry(res[("cross", name, m)]) for m in multipliers}}
            logger.info(f"[{b}] cross {name} @L{layer}: " + "  ".join(
                f"x{m:+g}={mean_p(res[('cross', name, m)]):.3f}" for m in multipliers))
        with open(out_path, "w") as f:   # after every behavior, so a crash keeps the rest
            json.dump(results, f)
    logger.info(f"Saved CAA A/B sweep to {out_path}")
    return results


def cosine_table(caa_raw: Dict[str, t.Tensor], ours: Dict[str, Tuple[int, t.Tensor]]) -> Dict:
    """cos(CAA vector, our direction) at the matched residual-stream point, and the
    best-matching CAA layer. ours: concept -> (our layer, vector)."""
    table = {}
    for concept, (our_layer, vec) in ours.items():
        u = vec.to(t.float64) / vec.to(t.float64).norm()
        row = {}
        for b, v in caa_raw.items():
            cos = (v / v.norm(dim=-1, keepdim=True)) @ u      # (n_layers,)
            matched = our_layer - 1
            row[b] = {"matched_caa_layer": matched,
                      "cos_matched": cos[matched].item() if 0 <= matched < len(cos) else None,
                      "best_caa_layer": int(cos.abs().argmax()),
                      "cos_best": cos[cos.abs().argmax()].item(),
                      "by_layer": cos.tolist()}
        table[concept] = row
    return table

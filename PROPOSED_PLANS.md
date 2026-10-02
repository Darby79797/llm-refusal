# Proposed Plans (2026-10-02)

Open next steps, roughly in priority order. Where things stand (RESULTS.md Summary): the refusal direction r̂ controls refusal on all 7 models, and the safety score replicates. Weight orthogonalisation is equivalent to ablation; it is nearly free on Llama and costs 5-20× a random edit on Qwen. Fine-tuning finds a separate, canonical off-switch for refusal (a direction that inhibits it without carrying it), and 32 refusal examples restore refusal after the edit. Empathy works on 6/7 models, hedging is a real negative, and sycophancy needs a response contrast.

Cost key: **S** = hours, no new heavy runs; **M** = a day or a 7-model sweep; **L** = new module + multi-day sweeps.

---

## 1. How few refusal examples regrow refusal? — M

32 refusal examples (~2% of the fine-tuning data) restore 82-100% refusal on every model and in both LoRA arms. Find the threshold: `--mode regrow --n-refusal-examples` 1, 2, 4, 8, 16, with 3 seeds each, on Llama-3-8B and Qwen2.5-7B (readers arm, where r̂ is unavailable). Also run benign-only to 1000 steps, to check that "benign fine-tuning doesn't bring refusal back" isn't just a 200-step artefact. This turns the durability finding into a number: how much refusal data undoes an abliteration.

## 2. What is the refusal inhibitor? — S/M

Rank-one fine-tuning reliably finds one write direction (the same on every seed) whose r̂-free part switches refusal off when written to. Ablating it from the model leaves refusal intact (RESULTS.md "Does Fine-Tuning Rediscover r̂?"). So it acts downstream of, or in parallel with, r̂. Questions:
- **What reads it?** Attribute the drop in refusal log-odds when writing along û⊥ to downstream heads and MLPs (`attribution.py`, as in the refusal-circuit analysis). Does the inhibition go through the components that write r̂, i.e. does writing û⊥ shrink r̂'s projection at later layers?
- **What is it?** Project harmful vs harmless prompts (and compliant vs refusing completions) onto û⊥ with `tools/project.py`. Is it a "this request is fine" feature that the model already uses, at low magnitude, on harmless prompts?
- **Is it general?** Does adding û⊥ also suppress the other concepts (empathy, hedging), or only refusal? Does the induce adapter's direction (|cos| 0.16-0.20 with the remove one) have the same structure in reverse?

## 3. Why does the edit cost Qwen 5-20× more than Llama? — M

The weight edit costs 0.06-0.19 nats of CE on Qwen2.5 (0.5B-7B), ≤0.04 on Llama, and the random-direction edit ≤0.03 everywhere. Candidates:
- r̂ on Qwen overlaps directions the model uses for ordinary text.
- The edit at every layer is more than refusal needs; Qwen loses nearly all refusal from single-layer ablation.

Measure CE for layer-restricted edits (only layers ≥ ℓ, or only ℓ±2) and for directions from other positions. The cheapest edit that still takes refusal to ~0% is the practically interesting one.

## 4. Is rank one enough? Rank-k orthogonalisation on the hard cases — M

`orthogonalize.py` already takes a [k, d] stack. Llama-2 is the outlier: single-layer ablation leaves 78.8% refusal, and global ablation leaves 5.1% (10% on JailbreakBench). Build a rank-k basis from the per-layer difference-in-means directions (top-k PCA of the L candidate r̂'s, or 2-3 depth-separated layers). Plot refusal, log-odds and CE against k for Llama-2, with Qwen2.5-7B as a control where k=1 already gives 0%. Apply the same to sycophancy (item 5).

## 5. Sycophancy from a response contrast — M

Take activations on the *same* prompts when the model was vs wasn't sycophantic (CAA-style answer contrast), instead of the prompt contrast, which encodes framing. On Qwen2.5-0.5B the behaviourally filtered direction works (42% → 5% ablated, 11% → 52% induced); the pipeline lacks a way to build it on models that rarely act sycophantically. Then test the "ablating refusal raises sycophancy" cross-concept finding against the text.

## 6. Category-specific directions and targeted edits — M

Compute difference-in-means per harm category (weapons, cyber, bio/chem, fraud, self-harm, harassment) and measure pairwise cosines with the global r̂. If they diverge, orthogonalise against *one* category's direction and measure refusal on each category, to test whether a weight edit can be targeted. The Llama-3 safety audit suggests harassment and misinformation behave differently (they were LlamaGuard's misses).

## 7. Jailbreaks through the lens of r̂ (Arditi §5) — M

Project known jailbreaks (roleplay, DAN-style, many-shot, and §5's adversarial suffixes) onto r̂ at the selected layer with `tools/project.py`. Test whether successful jailbreaks suppress the projection and whether the projection predicts success (AUROC). That would be a cheap refusal monitor, and it covers §5, the last unreplicated section of the paper.

## 8. Part 4: current open-weight models — L

The project's original goal (README Part 4). Run search → evaluate → safety → orthogonalise → regrow on current checkpoints: Qwen3 (4B/8B, thinking vs non-thinking mode, where the direction may sit at a different position), Gemma-3 (4B/12B) and Llama-3.2-3B. Questions: does it still work now, and is the edit still cheap and still undone by a few refusal examples on models with newer safety training?

Constraints:
- `orthogonalize.py` refuses Gemma-2/3: they normalise each sublayer's output before the residual add, so editing `o_proj`/`down_proj` doesn't remove r̂. Gemma needs hook ablation or a norm-aware edit.
- 12B-class models are borderline in bf16 on the M4 Pro; one model per process.

---

**Suggested order:** 2 first (cheap: mostly saved adapters plus a few forward passes) → 1 → 3 → 4, with 5-7 as independent threads and 8 once the rest is stable.

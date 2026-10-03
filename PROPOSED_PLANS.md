# Proposed Plans (2026-10-02)

Open next steps, roughly in priority order. Where things stand (RESULTS.md Summary): the refusal direction r̂ controls refusal on all 7 models, and the safety score replicates. Weight orthogonalisation is equivalent to ablation; it is nearly free on Llama and costs 5-20× a random edit on Qwen. Fine-tuning finds a separate, canonical off-switch for refusal (a direction that inhibits it without carrying it), and 32 refusal examples restore refusal after the edit. Empathy works on 6/7 models, hedging is a real negative, and sycophancy needs a response contrast.

Cost key: **S** = hours, no new heavy runs; **M** = a day or a 7-model sweep; **L** = new module + multi-day sweeps.

---

## 1. How few refusal examples regrow refusal? — M — **done 2026-10-02 on 0.5B (8-16 examples; tracks recent exposures), Llama-3-8B points running; see RESULTS.md Limb §2**

32 refusal examples (~2% of the fine-tuning data) restore 82-100% refusal on every model and in both LoRA arms. Find the threshold: `--mode regrow --n-refusal-examples` 1, 2, 4, 8, 16, with 3 seeds each, on Llama-3-8B and Qwen2.5-7B (readers arm, where r̂ is unavailable). Also run benign-only to 1000 steps, to check that "benign fine-tuning doesn't bring refusal back" isn't just a 200-step artefact. This turns the durability finding into a number: how much refusal data undoes an abliteration.

## 2. What reads the refusal inhibitor? — M — **open; the off-switch is now known to exist at every layer and to survive regrowth (Limb §2)**

Rank-one fine-tuning finds a canonical direction û⊥, orthogonal to r̂, that switches refusal off when written to. It isn't a harmfulness feature, it's refusal-specific (empathy unaffected), and it mostly bypasses r̂: later layers' r̂ falls by only a third, while refusal log-odds drop further than under r̂ ablation (RESULTS.md "Does Fine-Tuning Rediscover r̂?"). Open: what carries the effect from û⊥ to the refusal-token logits? Attribute the drop in refusal log-odds when writing û⊥ to downstream heads and MLPs (`attribution.py`, as in the refusal-circuit analysis). Do these overlap the components that write r̂, or form a separate path straight to the output?

## 3. Why does the edit cost Qwen 5-20× more than Llama? — M — **mostly answered 2026-10-02: Qwen's late MLPs write +r̂ into every prompt; the blocks after r̂'s layer carry 25-50% of the Alpaca/Pile cost (Limb §1, §3); the on-distribution part is still open**

The weight edit costs 0.06-0.19 nats of CE on Qwen2.5 (0.5B-7B), ≤0.04 on Llama, and the random-direction edit ≤0.03 everywhere. Candidates:
- r̂ on Qwen overlaps directions the model uses for ordinary text.
- The edit at every layer is more than refusal needs; Qwen loses nearly all refusal from single-layer ablation.

Measure CE for layer-restricted edits (only layers ≥ ℓ, or only ℓ±2) and for directions from other positions. The cheapest edit that still takes refusal to ~0% is the practically interesting one.

## 4. Is rank one enough? Rank-k orthogonalisation on the hard cases — M

`orthogonalize.py` already takes a [k, d] stack. Llama-2 is the outlier: single-layer ablation leaves 78.8% refusal, and global ablation leaves 5.1% (10% on JailbreakBench). Build a rank-k basis from the per-layer difference-in-means directions (top-k PCA of the L candidate r̂'s, or 2-3 depth-separated layers). Plot refusal, log-odds and CE against k for Llama-2, with Qwen2.5-7B as a control where k=1 already gives 0%. Apply the same to sycophancy (item 5).

## 5. Sycophancy from a response contrast — M — **done on 0.5B (induces up to 83%, doesn't ablate), 3B running; refusal→sycophancy judged real on all Qwens (Limb §4)**

Take activations on the *same* prompts when the model was vs wasn't sycophantic (CAA-style answer contrast), instead of the prompt contrast, which encodes framing. On Qwen2.5-0.5B the behaviourally filtered direction works (42% → 5% ablated, 11% → 52% induced); the pipeline lacks a way to build it on models that rarely act sycophantically. Then test the "ablating refusal raises sycophancy" cross-concept finding against the text.

## 6. Category-specific directions and targeted edits — M — **answered on 0.5B: one direction, no targeting possible (Limb §3); 3B/8B geometry queued**

Compute difference-in-means per harm category (weapons, cyber, bio/chem, fraud, self-harm, harassment) and measure pairwise cosines with the global r̂. If they diverge, orthogonalise against *one* category's direction and measure refusal on each category, to test whether a weight edit can be targeted. The Llama-3 safety audit suggests harassment and misinformation behave differently (they were LlamaGuard's misses).

## 7. Jailbreaks through the lens of r̂ (Arditi §5) — M — **done with hand-written templates on 0.5B/3B/Llama-3 (Limb §3); stronger templates for Llama-3 in progress; GCG needs network**

Project known jailbreaks (roleplay, DAN-style, many-shot, and §5's adversarial suffixes) onto r̂ at the selected layer with `tools/project.py`. Test whether successful jailbreaks suppress the projection and whether the projection predicts success (AUROC). That would be a cheap refusal monitor, and it covers §5, the last unreplicated section of the paper.

## 8. Part 4: current open-weight models — L

The project's original goal (README Part 4). Run search → evaluate → safety → orthogonalise → regrow on current checkpoints: Qwen3 (4B/8B, thinking vs non-thinking mode, where the direction may sit at a different position), Gemma-3 (4B/12B) and Llama-3.2-3B. Questions: does it still work now, and is the edit still cheap and still undone by a few refusal examples on models with newer safety training?

Constraints:
- `orthogonalize.py` refuses Gemma-2/3: they normalise each sublayer's output before the residual add, so editing `o_proj`/`down_proj` doesn't remove r̂. Gemma needs hook ablation or a norm-aware edit.
- 12B-class models are borderline in bf16 on the M4 Pro; one model per process.

---

**Suggested order:** 1 → 2 → 3 → 4, with 5-7 as independent threads and 8 once the rest is stable.

## 9. The edit up to r̂'s layer only — S/M (queued as tail jobs, 2026-10-02)

The edit-cost sweeps already show that orthogonalising r̂ out of only the blocks *before* its layer (blocks 0..D−1, embedding irrelevant) removes refusal completely on Qwen2.5-0.5B, 3B and Llama-3-8B at 20-60% of the full edit's CE (RESULTS.md "Limb Experiments" §1-3). Open, in priority order:
- the paper's safety score for that edit (JailbreakBench, 512 tokens, LlamaGuard 2) on Llama-3-8B, to confirm it is the same compliance as the full edit (queued: `600-safety-ltd-edit`);
- `--mode capability` with `--edit-layers`, for ARC/GSM8K/TruthfulQA on the restricted edit (needs capability.py to pass edit_layers through, a few lines);
- regrowth after the restricted edit: does leaving the late writers intact make refusal regrow faster or slower (`--mode regrow --orthogonalize-first --edit-layers …`, readers arm, 8 and 16 examples);
- the hook-ablation analogue (ablate at layers ≤ D only) as a sanity check that the weight and hook versions still agree when restricted.

## 10. The many-shot exception on Qwen2.5-3B — S — DONE 2026-10-03 (negative)

3B refuses 97% of many-shot-wrapped prompts although the r̂ projection at both decision positions (end-of-instruction token and generation boundary, read at L21's input) is near the level of the suppression template it complies with. Measured (`scripts/jailbreak_layers.py`, `results/jailbreak/Qwen2.5-3B-Instruct-refusal-layers.json`): no layer rebuilds the reading. Many-shot's last-token projection stays below plain's at every layer, and from L31 every prompt set (harmless and the complied suppression template included) rises together to +33 to +54 at L35, where many-shot (+48, refused) and suppression (+51, complied) are indistinguishable. That refusal runs through something other than r̂ (RESULTS.md §3). Open follow-up: what does mediate it (a search inside the many-shot-wrapped prompt set, or a probe at the template's "Assistant:" tokens).

## 11. Referee-proposed tests not yet run (2026-10-03)

From the five referee passes (results/analysis/referee-*.json). Queued: Llama-3 r16 at lr 1e-4 (is the 8-16 threshold an lr artefact?), the edited-model-without-regrowth and null-adapter controls for the off-switch overlap, random/clean L16 direction controls for the "latent axis", the OOD evaluation of the regrown Llama-3, the Qwen-7B regrown search, the outlier-masked r̂ edit cost and trajectory, and the jailbreak controls on 3B. Not yet queued, in rough priority:
- **Novel-target regrowth control** (`--regrow-examples-file`): harmful → a fixed non-refusal marker, or a harmless category → refusal, at n = 8/32. If it learns at the same exposure count, "capacity to refuse" is generic learnability.
- **Inhibitor mechanism**: ~~a `u_random` adapter variant (keep V, random U ⊥ r̂ at matched norm, 3-5 seeds)~~ done 2026-10-03 (3 seeds: 42 / 82 / 48% refusal vs 2% trained, ≈0 CE; RESULTS.md §1); still open: position-restricted application (prompt-only vs boundary-and-generation); prefill "I cannot" with the inhibitor installed. Distinguishes downstream inhibition from first-token steering.
- ~~**Generic-token trajectory**: per-layer r̂·x and norm over Pile tokens (sink excluded) on 0.5B/3B/Llama-3~~ done 2026-10-03 (`scripts/pile_trajectory.py`): the late +r̂ write is present at Pile tokens on Qwen (0.5B +6.1, 3B +48.9 at the last layer; Llama-3 ≤ +1.0) and disappears with the outlier coordinate zeroed (RESULTS.md §3). The outlier-masked edit cost and trajectory are also done (§1, §3).
- **The outlier coordinate** (new, 2026-10-03): on 0.5B zeroing r̂'s dim 490 keeps the edit's effect and removes 80% of its cost; on 3B zeroing dim 1874 breaks the edit (22% refusal, 15% false refusals). Next: the same on 1.5B (dims 1421/609) and 7B (458/2570); an edit of the outlier coordinate *only* (does it move refusal at all?); whether the masked 0.5B edit still complies on JailbreakBench at 512 tokens; a second direction seed.
- **Regrown Qwen-7B** (new): the cutoff-0.65 search strictly picks L17/P-1 but its ablation leaves 81%; the full-depth search is queued (`615-regrown100`). If it finds a late boundary mediator, 7B matches 0.5B and Llama-3; if not, 7B's regrown refusal is distributed.
- **Per-row CE** from `completion_ce` and an MLP-only (L17/L21/L22 down_proj) edit on 0.5B, plus mean-ablation instead of zero-ablation, for the edit-cost decomposition.
- **Token-matched trajectory positions** (`--positions`) on Qwen-3B vs Llama-3 ("assistant", "\n", `<|im_end|>`/`<|eot_id|>`).
- **XSTest safe split** per-layer AUROC (download needed): does r̂ probe refusal or topic?
- **Sycophancy_response on neutral and polarity-flipped prompts** with the validated judge: deference vs "affirm the proposition".
- **Category subset-noise null** (random 10-prompt subsets) and ablation of the category residuals c_k.
- Seeds: a second Llama-3 regrowth seed + regrown search (is L16/P-1 canonical?), 0.5B rank1 remove/null seeds 1-2 for the CE netting, random-direction edits with seeds 1-4.

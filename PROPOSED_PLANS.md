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
- ~~**Regrown Qwen-7B**~~ DONE 2026-10-03: the full-depth search finds L24/P-1, whose ablation takes regrown refusal 100 → 0%; 7B matches 0.5B and Llama-3. Open: its addition is 100% degenerate at the default scale, so a scale sweep would give the clean induction number.
- **Per-row CE** from `completion_ce` and an MLP-only (L17/L21/L22 down_proj) edit on 0.5B, plus mean-ablation instead of zero-ablation, for the edit-cost decomposition.
- **Token-matched trajectory positions** (`--positions`) on Qwen-3B vs Llama-3 ("assistant", "\n", `<|im_end|>`/`<|eot_id|>`).
- **XSTest safe split** per-layer AUROC (download needed): does r̂ probe refusal or topic?
- **Sycophancy_response on neutral and polarity-flipped prompts** with the validated judge: deference vs "affirm the proposition".
- **Category subset-noise null** (random 10-prompt subsets) and ablation of the category residuals c_k.
- Seeds: a second Llama-3 regrowth seed + regrown search (is L16/P-1 canonical?), 0.5B rank1 remove/null seeds 1-2 for the CE netting, random-direction edits with seeds 1-4.

## 12. Does the regrown model still obey the original r̂? — S — **DONE 2026-10-04 on 0.5B, Llama-3.2-1B, Llama-3-8B, Qwen-7B (RESULTS.md Limb §6): yes, more than a benign fine-tune does**

**Question.** After r̂ is edited out and refusal is fine-tuned back along a new axis, does adding the clean r̂
to harmless prompts still trigger refusal, or did fine-tuning train that response away? Script:
`scripts/rhat_in_regrown.py` (one model load; variants clean / edited / edited + each regrow arm; per-prompt
log-odds, degenerate rate and samples saved to `results/finetune/<model>-rhat-in-regrown.json`).

**Comparison that counts.** Regrown (32 refusal examples) against the *benign-only* fine-tune of the same arm,
in log-odds *shift* (r̂ added minus nothing, per prompt). Not against the edited model: benign fine-tuning alone
moves the harmless baseline ~3.7 nats away from refusal (0.5B), which drops r̂-induced refusal 90% → 1% while
r̂'s push shrinks only from +6.1 to +4.7 nats. Rates mislead here; shifts don't.

**Qwen2.5-0.5B (one seed per arm, 80 harmless prompts, r̂ at L14 strength 1).**

| Variant | Refusal | Log-odds none → r̂ | Shift |
|---|---|---|---|
| clean | 100% | −3.6 → +1.7 | +5.2 |
| edited | 90% | −5.1 → +1.0 | +6.1 |
| readers, benign only | 1% | −8.8 → −4.1 | +4.7 |
| readers, 32 refusals | 14% | −7.3 → −2.3 | +5.0 |
| writers, benign only | 1% | −8.7 → −4.5 | +4.2 |
| writers, 32 refusals | 35% | −6.9 → −1.0 | +5.9 |

Refusal training does not remove the response to r̂ (shift ≥ the benign-only one in both arms); the regrown model
has two triggers. Ablating the regrown L18 mediator trims r̂'s effect only a little (14 → 9%, 35 → 26%; clean
100 → 96%). Random norm-matched direction: 0-9%. Strength 4 and every-layer addition: degenerate, not results.

**Outcome at 1B-8B (2026-10-04).** Regrown-minus-benign shift, per arm/seed: Llama-3.2-1B +3.3 / +4.1 (×1);
Llama-3-8B +4.9 / +1.2 / +5.9 readers seeds 0-2, +7.3 writers (×1); Qwen-7B +6.6 / +4.3 / +4.7, +5.5 (×0.5; ×1
saturates). Paired per-prompt SD 1-2 nats (tighter than the 3.5-4 assumed above); seed spread ±2.5 nats on Llama-3
(larger than the ±1-1.5 assumed). The design's rule had no "amplified" branch; every estimate lands there. Benign
fine-tuning alone shrinks r̂'s push 2-4×, refusal training restores part of it. Mediator-ablated r̂: 0-14% on 7B and
1B variants, but largely intact on Llama-3-8B, where the mediator used (seed-0 L16) has since been superseded by the
full-depth L30 pick.

**Llama-3.2-1B (first pass, 2026-10-03: clean and edited).** Regrowth replicates (32
examples: 100% harmful refusal in both arms; benign only: 0%); the regrown mediator is L14/P-1 of 16. r̂ at L8
strength 1 induces 76% (clean) / 80% (edited), shift ≈ +16 nats, per-prompt SD of the shift 3.5-3.9. **Unlike
0.5B, ablating the L14 mediator abolishes r̂'s induction in the clean model** (76 → 0%, log-odds +4.5 → −5.8):
on Llama-3.2-1B r̂ acts through the late boundary direction even before any editing.

### Design for Qwen2.5-7B and Llama-3-8B

Adapters exist for both (readers/writers × r0/r32, seed 0); mediators: Llama-3 `results/regrown-Meta-Llama-3-8B-Instruct-refusal-direction`
(L16/P-1), 7B `results/regrown100-Qwen2.5-7B-Instruct-refusal-direction` (L24/P-1).

*Power, by reasoning rather than formal tests.* The unit that varies most is the adapter, not the prompt.
- **Prompts.** The per-prompt SD of the shift is ~3.5-4 nats, and paired differences between variants on the
  same prompts should be no larger. With all 80 harmless eval prompts, a ~1.2-nat difference in mean shift is
  resolvable. "Trained away" means a multi-nat drop, so 80 prompts are plenty, and more would not help.
- **Adapters.** On 0.5B the two arms disagree by ~1.4 nats on the regrowth effect (+0.3 vs +1.7), so one
  adapter seed has about ±1-1.5 nats of its own noise. One seed per arm therefore settles only large effects.
- **Rule.** Phase 1 runs one seed, both arms. Stop there if the regrown-vs-benign shift difference is either
  within ±1 nat in both arms ("preserved") or below −3 nats in both arms ("trained away"). If it lands in
  between, or the arms disagree in sign, run Phase 2: train two more seeds of readers r0 and r32 (~20 min each
  on 8B) and rerun those variants only. Three seeds put the seed-mean noise at ~±0.7 nats.
- **Rates.** Detection rates are secondary. Report them only at strengths where the edited model sits at 50-90%
  refusal, because 100% hides differences and degenerate text fakes them. Batch-size noise is a few pp, so no
  rate difference under ~10 pp counts.

*Conditions per variant* (trimmed from the small-model run): harmful none, harmless none, r̂ at its layer
×0.25/0.5/1 (8B models saturate at ×1: clean addition 97.5-100%), random norm-matched ×1, mediator added ×1
(positive control), mediator ablated + r̂ ×1, mediator projections. Drop ×2/×4 and every-layer
addition, which were degenerate on the small models. That is about 9 generation passes per variant.

*Cost.* An 8B pass over 80 prompts is ~1-1.5 min, so 6 variants × 9 conditions is ~1-1.5 h per model, plus
~1.5 h for Phase 2 if needed. Run one model at a time and never alongside another GPU job: two OOM crashes on
2026-10-03 came from sharing the GPU.

    .venv/bin/python3 llm-refusal/scripts/rhat_in_regrown.py --model meta-llama/Meta-Llama-3-8B-Instruct \
      --strengths 0.25,0.5,1 --mediator results/regrown-Meta-Llama-3-8B-Instruct-refusal-direction
    .venv/bin/python3 llm-refusal/scripts/rhat_in_regrown.py --model Qwen/Qwen2.5-7B-Instruct \
      --strengths 0.25,0.5,1 --mediator results/regrown100-Qwen2.5-7B-Instruct-refusal-direction

(The script's random ×max-strength and every-layer conditions follow `--strengths`; with max 1 they cost one
extra pass. The mediator-ablated condition runs at ×1 and ×max.)

**What each outcome means.** If the shift is preserved at 8B, removing r̂ and regrowing refusal leaves two
triggers at every scale, so an r̂-based monitor still fires on the regrown model. If it is trained away at 8B
only, scale changes how fine-tuning treats an unused input direction. If ablating the mediator kills r̂'s
induction on Llama-3-8B, as on Llama-3.2-1B, then in the Llama family r̂ is upstream of the late boundary
direction even in the clean model, which explains why regrowth reuses that direction.

## 13. Follow-ups from the 2026-10-03/04 queue — S each (RESULTS.md Limb §6) — DONE 2026-10-04 (jobs 200-250, RESULTS.md §7) except the masked-edit JailbreakBench run (now 14d). Outcome: r̂ sensitivity from refusal training does not track regrowth; on Llama-3 it goes through the regrown L29-30 axis too (no bypass); the 7B regrown axis transfers across seeds, Llama-3's only from L29; the clean 7B needs that axis, the clean Llama-3 does not; the outlier cost drop is specific to those coordinates

- **Llama-3-8B r̂ bypass against the real mediator**: rerun the mediator-ablated conditions of plan 12 with
  `results/regrown100-Meta-Llama-3-8B-Instruct-refusal-direction` (L30/P-1) instead of L16. Plan 12's "r̂ bypasses
  the mediator on Llama-3-8B" rests on the superseded L16 pick (`rhat_in_regrown.py --mediator ...`; ~1 h).
- **Cross-seed transfer**: ablate seed 0's regrown direction inside the seed-1/2 regrown models (and vice versa). The
  same-layer regrowth deltas agree at cos 0.6-0.86; does ablating one seed's axis remove another seed's refusal?
- **Is Llama-3-8B's L30 axis needed by the clean model**, as 7B's L24 is (89 → 4%)? `--direction-file
  results/regrown100-Meta-Llama-3-8B-Instruct-refusal-direction` evaluate on the clean model, plus `axis_vs_clean.py`.
- **Masked 7B edit at 512 tokens / JailbreakBench** (needs the local LlamaGuard via Ollama): does the near-free
  masked edit comply as fully as the plain one?
- **Seed 1 of Llama-3 readers r32** has 42% degenerate harmful refusals and the weakest r̂ response (+1.2): eyeball
  its generations before pooling it.

## 14. Follow-ups from the plan-13 checks — S each (RESULTS.md §7) — queued 2026-10-04 as jobs 300-330

- **14a. Llama-3 cross-seed asymmetry: layer or seed?** (job 300, `scripts/seed_direction_stems.py` +
  `ablate_in_variants.py --tag crossseed-delta`). Seed 0's L30 direction leaves 60% of seeds 1-2's refusal, their
  L29 directions remove everything. Ablate seed 0's raw L29 contrast, each seed's regrowth delta (minus the benign
  twin) at L29/L30, and random, inside all three regrown models. If seed 0's L29 contrast or delta transfers, the gap
  is depth; if only the deltas transfer, seed 0's direction carries a seed-specific part that seeds 1-2 don't use.
- **14b. Does ablating a regrown axis cause false refusal?** (job 310, `--harmless-gen`). Harmless first-token
  log-odds rise 4-7 nats under ablation. Generate on the harmless prompts in the clean and regrown Llama-3 and 7B
  with the regrown axis, random and r̂ ablated. A real false-refusal rate means the ablation results above carry a
  disruption cost; ~0% means the log-odds move is a flatter first token.
- **14c. Is Llama-3's L30 axis a trigger in the clean model?** (job 320). It is not needed there (100 → 99%). Add it
  to harmless prompts at ×0.25 / 0.5 / 1, as for 7B's L24 (0 / 17.5 / 100%) and the old L16 (80% at ×1).
- **14d. Masked 7B edit on JailbreakBench** (job 330, 512 tokens, LlamaGuard 2 via the local Ollama). Plain r̂ edit
  vs the outlier-masked edit: does the near-free edit comply as fully?

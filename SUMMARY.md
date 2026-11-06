# Summary

Residual-stream directions that mediate refusal and related behaviours, extending Arditi et al. to eight models: Qwen2.5 0.5B/1.5B/3B/7B, Llama-2-7B, Llama-3-8B, Llama-3.1-8B and Llama-3.2-1B. Full numbers, methods and caveats are in [RESULTS.md](RESULTS.md); section references (§) point there.

## 1. Arditi replicates on every model

- One refusal direction r̂ per model. Ablating it takes refusal from 89-100% to 0-5%; adding it at one layer makes 97.5-100% of harmless prompts refused, with fluent output both ways (Llama-3.2-1B: 80% → 1%, induction 76%).
- Ablation takes LlamaGuard-2-unsafe output on JailbreakBench from 1-18% to 66-89%; on Llama-3-8B 2% → 83%, against the paper's 3% → 85%.
- Weight orthogonalisation reproduces hook ablation. It costs ≤0.04 nats of CE on the Llamas and 0.06-0.19 on the Qwens (3× a random edit on 0.5B, 5× or more on the larger Qwens), on 0.5B, 1.5B and 7B almost all of it through one or two outlier coordinates.
- Llama-2 is the exception for single-layer ablation, which leaves 79% refusal: its refusal is spread across layers.

## 2. Fine-tuning finds other routes

- A rank-one adapter trained to remove refusal finds its own direction (|cos| 0.24-0.41 with r̂), the same across seeds; it inhibits refusal downstream rather than carrying it.
- Removal doesn't last: after the edit, 8-16 refusal examples (~1% of the data) bring refusal back, and 32 restore 82-100%, through a new direction at 75-96% depth.
- Removing each regrown direction in turn gives a fresh orthogonal one every round, regrowing more slowly and at rising CE cost.

## 3. Late layers and the answer

- On the Qwens a late stage after r̂ turns harm recognition into a refusal: with it ablated, 40-73% of answers become impersonal moralising rather than compliance. Llama-3.1 computes it but doesn't need it (73-93% refusal remains).
- Refusal is decided at the prompt: r̂ added on the prompt tokens alone induces 91-100% refusal on all eight models, and the answer carries it through attention.
- Qwen2.5-3B, 7B, 1.5B and Llama-3.2-1B also re-read refusal along the answer, in a narrow window 2-6 layers past r̂. Removing that direction from the generated tokens only cuts refusal by 43-77pp on the 7B, 1.5B and 1B, and on the 3B takes JailbreakBench refusal 87% → 20%; with r̂ also taken off the prompt, the 3B is as jailbroken as under full ablation (68% unsafe).
- Latest (§101-§105): adding a refusal direction to the answer alone induces refusal in a band just past r̂ on most models, including the large Llamas, which have no answer-side necessity (Llama-2 90%, Llama-3-8B 61%, Llama-3.1 48%). The unknown direction added the same way writes "I don't know" over facts the model has ("'Jane Eyre' is a novel by Charlotte Brontë, but I couldn't find any information on this book"); the limitation direction doesn't induce "I can't" cleanly.

## 4. Other concepts

- Ablating refusal raises sycophancy on every Qwen (+18 to +60pp); a direction from a response contrast can induce it. Empathy has a working direction on 6/7 models; hedging is a real negative (these models don't hedge on the test prompts).
- Inability, realtime, memory, persona and feelings form one "I'm an AI and can't…" switch. The mean of the first three, subtracted, removes all three limits' claims (0-13%) and, added, induces them (57-98%) on six models; it transfers to French (0-3% / 46-90%).
- The same push jailbreaks: 43-69% of JailbreakBench answers unsafe on five of seven models (random push 1-10%), mostly still with r̂ projected out.
- "Unknown" (invented entities) has its own familiarity-ordered direction; with it removed, every model answers invented questions with confident inventions (60-97%).

Most rates come from about 100 prompts (20-40 for most concept sets), so differences of a few points are noise. Refusal rates are phrase-matched; where an intervention turns refusal into moralising, LlamaGuard 2 labels are used instead.

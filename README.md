A project to reproduce and extend Arditi and Obeso's mechanistic interpretability paper "Refusal in Language Models is Mediated by a Single Direction". Written from the paper only, own code, own prompts, on the models people actually run in 2026.

Docs: [SUMMARY.md](SUMMARY.md) (short version) · [RESULTS.md](RESULTS.md) (all the numbers) · [ARCHITECTURE.md](ARCHITECTURE.md) (module map, adding a model/concept) · [ARDITI.md](ARDITI.md) (replication status) · [DECISION_COMPLEXITY.md](DECISION_COMPLEXITY.md) (design tradeoffs) · [PROPOSED_PLANS.md](PROPOSED_PLANS.md) (the plan log, 160-odd numbered plans) · [future_plans.md](future_plans.md), [BACKLOG.md](BACKLOG.md) (parked)

# The original plan

1. Reproduce Arditi on current models.
2. Extend from "just refusal" to refusal plus other concepts.
3. Reproduce Panickssery et al.'s "Steering Llama 2 via Contrastive Activation Addition" too - different method, specific model, sycophancy and a few other features.
4. Combine the two and apply to state of the art open-weight models. Instead of asking "did these papers work at the time" (which is going to be yes, papers that release all their code tend to reproduce) ask "do they work now" (strongly expected) and "does the content transfer into other domains now" (weakly expected). This was the actual reason for doing any of it.

# What got done

Part 1, all of it bar §5 (adversarial suffixes, never attempted), on eight models - four Qwen2.5 sizes, Llama-2-7B, both Llama-3 8Bs, Llama-3.2-1B.

Part 2 drifted. Empathy works, hedging doesn't exist on these models, sycophancy needed a contrast on the answers rather than the prompts. The concepts that ended up mattering weren't on the list at all: they're the model talking about itself. Can't do that, can't see that, don't remember that, don't know that. A lot of the later plans are about those.

Part 3, half. The A/B replication holds. The open-ended half is stuck on the judge - the local one scores a refusal as compliance and the paper used GPT-4.

Part 4, half, and not the interesting half. Everything replicates on the 2024 checkpoints (Qwen2.5, Llama-3.1/3.2). The 2026 ones (Qwen3, Gemma-3, Llama-3.2-3B) never got downloaded. Transfer across methods: the two papers' refusal directions aren't the same direction, so that answer is messier than "yes".

Where most of October went: a question not in the plan. Once the direction has removed refusal, what is the model still doing? Fine-tuning around the edit, then the layers after the direction, then which tokens of the prompt and answer refusal actually sits in. That's SUMMARY §2-3 and about a hundred RESULTS sections.

# Next

- Plans 159-160 are queued (long answers, Llama-2's unknown direction). Finish those.
- Download the 2026 checkpoints and point the pipeline at them. This is the part 4 I said I'd do and it should just be a `--model` flag.
- A judge that can tell a refusal from an answer, otherwise the CAA open-ended half and hedging/opinion stay inconclusive.
- 13B+ for sycophancy; moving a direction between model families. Both in future_plans.md, neither started.

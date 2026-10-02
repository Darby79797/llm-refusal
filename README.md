A project to reproduce and extend Arditi and Obeso's mechanistic interpretability paper "Refusal in Language Models is Mediated by a Single Direction"

Docs: [ARCHITECTURE.md](ARCHITECTURE.md) (module map, extension guides) · [RESULTS.md](RESULTS.md) (current numbers and findings) · [ARDITI.md](ARDITI.md) (replication status) · [DECISION_COMPLEXITY.md](DECISION_COMPLEXITY.md) (design tradeoffs) · [future_plans.md](future_plans.md) (roadmap)

# Part 1:
- done for §2-4: the refusal direction, refusal and safety scores on all 7 models, and weight orthogonalisation (equivalent to ablation; nearly free on Llama, costly on Qwen; reversed by a few refusal examples of fine-tuning). §5 (adversarial suffixes) not attempted. Headline numbers: the Summary at the top of [RESULTS.md](RESULTS.md).

Reproduce Arditi's paper, using only the paper and writing my own code, using my own prompts/data, and using updated evaluations of the models from 2026.

# Part 2: 
- done for empathy (works on 6/7 models), hedging (real negative) and sycophancy (needs a response-contrast direction), plus cross-concept interference (directions close to independent).

Extend this from "just refusal" to "refusal, and also other concepts".

# Part 3: 
- A/B half done: all 7 behaviours steer on Llama-2-7B, peaking at layers 11-13 as in the paper. On the other 6 models steering needs ×2; refusal moves on all of them, sycophancy weakly (not at all on Qwen2.5-0.5B). Open-ended half pending an LLM-judge API key.

Also reproduce Panickssery et al.'s "Steering Llama 2 via Contrastive Activation Addition", which uses a different method to do causal interventions on a specific model, for sycophancy and other features.

# Part 4: 
- the original reason for this research project

Combine the methodology of these two papers, and apply to state of the art open-weight models. Instead of asking the question "did these papers work at the time" (which is going to be yes, papers that release all their code tend to reproduce) with "do these papers work now" (strongly expected), and "do these papers' content transfer into the other domains now" (weakly expected).
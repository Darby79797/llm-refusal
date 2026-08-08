A project to reproduce and extend Arditi and Obeso's mechanistic interpretability paper "Refusal in Language Models is Mediated by a Single Direction"

Docs: [ARCHITECTURE.md](ARCHITECTURE.md) (module map, extension guides) · [RESULTS.md](RESULTS.md) (current numbers and findings) · [ARDITI.md](ARDITI.md) (replication status) · [DECISION_COMPLEXITY.md](DECISION_COMPLEXITY.md) (design tradeoffs) · [future_plans.md](future_plans.md) (roadmap)

# Part 1:
- mostly completed 

Reproduce Arditi's paper, using only the paper and writing my own code, using my own prompts/data, and using updated evaluations of the models from 2026.

# Part 2: 
- mostly completed

Extend this from "just refusal" to "refusal, and also other concepts".

# Part 3: 
- not yet started

Also reproduce Panickssery et al.'s "Steering Llama 2 via Contrastive Activation Addition", which uses a different method to do causal interventions on a specific model, for sycophancy and other features.

# Part 4: 
- the original reason for this research project

Combine the methodology of these two papers, and apply to state of the art open-weight models. Instead of asking the question "did these papers work at the time" (which is going to be yes, papers that release all their code tend to reproduce) with "do these papers work now" (strongly expected), and "do these papers' content transfer into the other domains now" (weakly expected).
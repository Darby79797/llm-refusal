# Backlog

Low-priority items parked out of PROPOSED_PLANS.md. Not scheduled; pick up only when asked.

## Measurement hygiene (was plan 27, 2026-10-12)

LG2 labels at 160 tokens now carry the conclusions; the log-odds
token fix matters only for Llama's harmless log-odds, and `--harmless-gen` already sidesteps it. Re-labelling: jobs
before 1077 kept no texts, so it means re-running plans 18-20 for numbers §12-13 supersede. One item is worth
adding: a degenerate label in `label_deflection.py` (16-19% of 0.5B PC1+3 / top-3 answers are degenerate and land in
"other").

- Done 2026-10-12: moralise/other split in `label_deflection.py`; `--max-new-tokens` in `ablate_in_variants.py`
  (jobs call the labeller as their last step instead of an `--label` option).
- A refusal-specific log-odds token set without bare "I"/"As" reported next to the current one (the Llama harmless
  artifact, §12). Report both rather than changing the old metric, so the earlier tables stay comparable.
- Re-label plans 18-20's saved texts where they exist (only jobs ≥1077 kept every response; earlier ones would need a
  re-run).

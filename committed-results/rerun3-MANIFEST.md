# Post-fix re-evaluation sweep

Generation path fixed (generate_with_hooks); log_odds_metric reported per condition.
One row appended per model as it completes.

Batch size is fixed *within* each run (all six conditions share it) but varies by
row, and is recorded per row: bs=8 for the fp32 models and Llama-3-8B, bs=2 for the
remaining 7-8B models, where bf16 at bs=8 exceeded this machine's memory. In bf16 a
rate can shift a few points across batch sizes, so do not compare rows at
single-point precision; fp32 is exactly batch-invariant. See RESULTS.md.

| model | dtype | bs | baseline | global abl | layer abl | layer add | log-odds base -> global abl | min |
|---|---|---|---|---|---|---|---|---|
| Qwen2.5-0.5B-Instruct | float32 | 8 | 88.9% | 1.0% | 5.1% | 98.8% | +0.60 -> -4.21 | 3 |
| Qwen2.5-1.5B-Instruct | float32 | 8 | 92.9% | 0.0% | 2.0% | 100.0% | +1.18 -> -4.66 | 6 |
| Qwen2.5-3B-Instruct | float32 | 8 | 96.0% | 0.0% | 4.0% | 100.0% | +4.07 -> -8.15 | 12 |
| Meta-Llama-3-8B-Instruct | auto | 8 | 100.0% | 0.0% | 6.1% | 97.5% | +9.79 -> -10.60 | 357 |
| Llama-3.1-8B-Instruct | auto | 2 | 95.0% | 0.0% | 2.0% | 83.8% | +7.88 -> -11.53 | 54 |
| Qwen2.5-7B-Instruct | auto | 2 | 88.9% | 0.0% | 0.0% | 100.0% | +4.20 -> -14.56 | 52 |
| Llama-2-7b-chat-hf | auto | 2 | 100.0% | 5.1% | 78.8% | 97.5% | +8.92 -> -6.91 | 48 |

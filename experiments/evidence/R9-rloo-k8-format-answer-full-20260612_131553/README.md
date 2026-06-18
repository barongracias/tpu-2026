# R9 RLOO K=8 Format-Then-Answer Evidence

Run root:
`/home/fredlawrence/tpu-runs/part-i/R9-rloo-k8-format-answer-full-20260612_131553`

This directory contains lightweight report evidence copied from the run root:
per-prompt greedy eval CSVs, stage metadata, launch commit, and summary text.
It intentionally excludes checkpoint trees and TensorBoard event files.
It also includes compact TensorBoard scalar exports:
`tensorboard_scalars.csv` and `tensorboard_scalar_summary.json`.

## Provenance

- Branch/commit: `fred`, `a93ea9f387b8c5083607263daf157929482c50cb`
- Estimator/K/seed: `ADV_ESTIMATOR=rloo`, `NUM_GENERATIONS=8`, `RUN_SEED=0`
- Data: full GSM8K TFDS training data, not the hard-medium manifest
- Eval: shared `gsm8k_test_seed0_n64.jsonl`, `EVAL_SEED=0`
- Stage 1: `REWARD_PROFILE=format_heavy`, step `500`
- Stage 2: `REWARD_PROFILE=answer_heavy`, through step `3364`

## Greedy Eval Summary

| Eval | Restored step | Exact | Partial | Format | Empty |
| --- | ---: | ---: | ---: | ---: | ---: |
| `base_greedy.csv` | none | 31/64 (48.44%) | 31/64 (48.44%) | 1/64 (1.56%) | 0/64 |
| `stage1_step500_greedy.csv` | 500 | 30/64 (46.88%) | 36/64 (56.25%) | 60/64 (93.75%) | 0/64 |
| `stage2_step500_greedy.csv` | 500 | 30/64 (46.88%) | 36/64 (56.25%) | 60/64 (93.75%) | 0/64 |
| `stage2_step1000_greedy.csv` | 1000 | 30/64 (46.88%) | 31/64 (48.44%) | 62/64 (96.88%) | 0/64 |
| `stage2_step2000_greedy.csv` | 2000 | 33/64 (51.56%) | 35/64 (54.69%) | 56/64 (87.50%) | 0/64 |
| `stage2_step2500_greedy.csv` | 2500 | 37/64 (57.81%) | 37/64 (57.81%) | 58/64 (90.62%) | 0/64 |
| `stage2_step3000_greedy.csv` | 3000 | 37/64 (57.81%) | 38/64 (59.38%) | 57/64 (89.06%) | 0/64 |
| `stage2_step3364_greedy.csv` | 3364 | 34/64 (53.12%) | 36/64 (56.25%) | 56/64 (87.50%) | 0/64 |

Report-use caveat: best retained exact is tied at steps `2500` and `3000`;
the final checkpoint is step `3364` and is lower. Use paired/bootstrap
uncertainty before making a strong improvement claim from this 64-prompt eval.

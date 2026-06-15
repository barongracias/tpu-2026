# Evidence: Harvey GRPO K=16 full-test eval, 2026-06-15

## Scope

- Run id: `H-grpo-k16-cap2500-s0-20260614-r2`
- Method: GRPO
- K: `16`
- Seed: `0`
- Cap: `2500` max training steps; final is reported as last-good retained checkpoint.
- Checkpoint policy: step `500` selected from the n=64 screen; step `500` is the final / last-good checkpoint.
- Training status: stopped before the cap with `RESOURCE_EXHAUSTED` after scalar step `678`; last retained checkpoint is step `500`.

## Manifest

Screening manifest:

- `/home/harvey/tpu-runs/part-i/manifests/gsm8k_test_seed0_n64.jsonl`
- SHA-256: `9aa1814e295e155f4735c9302a7f9c28c6aaa2319074e053a5888e3f3b2448b0`
- Rows: `64`

Full-test manifest:

- `experiments/manifests/gsm8k_test_seed0_full.jsonl`
- SHA-256: `07f0f0fc10580dee941b6f921a3986854a8b0b74529d9bb952662d5daaea6bb2`
- Rows: `1,319`

## Artefacts

Committed lightweight artefacts are staged under:

- `experiments/evidence/H-grpo-k16-cap2500-s0-20260614-r2/`

Raw TPU-side logs remain under:

- `/home/harvey/tpu-runs/part-i/H-grpo-k16-cap2500-s0-20260614-r2/logs/auto_eval_k16_after_train_20260615.log`

## n=64 Screening Result

See:

- `/home/harvey/tpu-runs/part-i/H-grpo-k16-cap2500-s0-20260614-r2/eval/H-grpo-k16-cap2500-s0-20260614-r2_n64_screen_summary.txt`
- `/home/harvey/tpu-runs/part-i/H-grpo-k16-cap2500-s0-20260614-r2/eval/H-grpo-k16-cap2500-s0-20260614-r2_n64_screen_metrics.csv`

Selected best retained checkpoint: step `500`.

## Full-Test Results

| Model / checkpoint | Exact | Partial | Format | Empty |
|---|---:|---:|---:|---:|
| Base greedy | 625/1319 (47.38%) | 659/1319 (49.96%) | 53/1319 (4.02%) | 0/1319 |
| K=16 step 500 | 703/1319 (53.30%) | 729/1319 (55.27%) | 1100/1319 (83.40%) | 0/1319 |
| K=16 step 500 | 703/1319 (53.30%) | 729/1319 (55.27%) | 1100/1319 (83.40%) | 0/1319 |

Bootstrap confidence intervals used `10000` resamples and seed `12345`.

| Model / comparison | Result |
|---|---:|
| Base exact CI | 625/1319 (47.38%), 95% CI [44.73%, 50.11%] |
| K=16 best exact CI | 703/1319 (53.30%), 95% CI [50.57%, 56.03%] |
| K=16 final exact CI | 703/1319 (53.30%), 95% CI [50.57%, 55.95%] |
| K=16 best vs base | +5.91 pp [+3.18 pp, +8.64 pp] (CI excludes 0) |
| K=16 final vs base | +5.91 pp [+3.18 pp, +8.64 pp] (CI excludes 0) |
| K=16 best vs final | +0.00 pp [+0.00 pp, +0.00 pp] (CI includes 0) |

## Interpretation

Pending report collation. This is the capped K=16 point and should be labelled by its actual final / last-good step.

# Evidence: Harvey GRPO K=4 full-test eval, 2026-06-14

## Scope

This note records the runbook-compliant full GSM8K test-set confirmation eval for Harvey's GRPO K=4 run:

- Run id: `H-grpo-k4-full-s0-20260614`
- Commit: `57c6409add0d81bbdb32ca7f4b3e176b4e044068`
- Method: GRPO
- K: `4`
- Seed: `0`
- Checkpoint policy: step `2750` was selected as the best retained checkpoint from the n=64 screen; step `3364` is the final checkpoint.

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

- `experiments/evidence/H-grpo-k4-full-s0-20260614/`

Raw TPU-side logs remain under:

- `/home/harvey/tpu-runs/part-i/H-grpo-k4-full-s0-20260614/logs/eval_k4_runbook_20260614.log`

## n=64 Screening Result

See:

- `/home/harvey/tpu-runs/part-i/H-grpo-k4-full-s0-20260614/eval/H-grpo-k4-full-s0-20260614_n64_screen_summary.txt`
- `/home/harvey/tpu-runs/part-i/H-grpo-k4-full-s0-20260614/eval/H-grpo-k4-full-s0-20260614_n64_screen_metrics.csv`

Selected best retained checkpoint: step `2750`.

## Full-Test Results

| Model / checkpoint | Exact | Partial | Format | Empty |
|---|---:|---:|---:|---:|
| Base greedy | 625/1319 (47.38%) | 659/1319 (49.96%) | 53/1319 (4.02%) | 0/1319 |
| K=4 step 2750 | 682/1319 (51.71%) | 712/1319 (53.98%) | 1104/1319 (83.70%) | 0/1319 |
| K=4 step 3364 | 681/1319 (51.63%) | 712/1319 (53.98%) | 1097/1319 (83.17%) | 0/1319 |

Bootstrap confidence intervals used `10000` resamples and seed `12345`.

| Model / comparison | Result |
|---|---:|
| Base exact CI | 625/1319 (47.38%), 95% CI [44.73%, 50.11%] |
| K=4 best exact CI | 682/1319 (51.71%), 95% CI [49.05%, 54.36%] |
| K=4 final exact CI | 681/1319 (51.63%), 95% CI [49.05%, 54.36%] |
| K=4 best vs base | +4.32 pp [+1.36 pp, +7.20 pp] (CI excludes 0) |
| K=4 final vs base | +4.25 pp [+1.67 pp, +6.90 pp] (CI excludes 0) |
| K=4 best vs final | +0.08 pp [-1.97 pp, +2.05 pp] (CI includes 0) |

## Interpretation

Pending report collation. The checkpoint-selection caveat from the full-test eval runbook applies: the best retained checkpoint was selected on the n=64 screen before full-test confirmation.

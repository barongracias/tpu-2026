# H GRPO K=4 evidence

Run id: `H-grpo-k4-full-s0-20260614`

Run root:

`/home/harvey/tpu-runs/part-i/H-grpo-k4-full-s0-20260614`

## Full-test eval status

Completed against the shared full GSM8K test manifest:

- Manifest: `experiments/manifests/gsm8k_test_seed0_full.jsonl`
- SHA-256: `07f0f0fc10580dee941b6f921a3986854a8b0b74529d9bb952662d5daaea6bb2`
- Rows: `1,319`

The completed full-eval artefacts are:

- `eval/base_full.csv`
- `eval/H-grpo-k4-full-s0-20260614_step2750_full.csv`
- `eval/H-grpo-k4-full-s0-20260614_step3364_full.csv`
- `eval/H-grpo-k4-full-s0-20260614_full_ci.csv`
- `eval/H-grpo-k4-full-s0-20260614_full_ci.json`
- `metadata/run_metadata.json`

## Results

| Model / checkpoint | Exact | Partial | Format | Empty |
|---|---:|---:|---:|---:|
| Base greedy | 625/1319 (47.38%) | 659/1319 (49.96%) | 53/1319 (4.02%) | 0/1319 |
| GRPO K=4 step 2750 | 682/1319 (51.71%) | 712/1319 (53.98%) | 1104/1319 (83.70%) | 0/1319 |
| GRPO K=4 step 3364 | 681/1319 (51.63%) | 712/1319 (53.98%) | 1097/1319 (83.17%) | 0/1319 |

Bootstrap confidence intervals used `10000` resamples and seed `12345`.

| Comparison | Delta exact | 95% CI | Result |
|---|---:|---:|---|
| K=4 step 2750 vs base | +4.32 pp | [+1.36 pp, +7.20 pp] | CI excludes 0 |
| K=4 step 3364 vs base | +4.25 pp | [+1.67 pp, +6.90 pp] | CI excludes 0 |

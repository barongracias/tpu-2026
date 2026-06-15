# R7 RLOO K=8 deterministic evidence

Run id: `R7-rloo-k8-det-harvey-full-s0-20260611_105132`

Run root:

`/home/ext_harveybermingham1_gmail_com/tpu-runs/part-i/R7-rloo-k8-det-harvey-full-s0-20260611_105132`

## Full-test eval status

Completed against the shared full GSM8K test manifest:

- Base and selected step `2000`: 2026-06-14 UTC
- Final step `3364`: 2026-06-15 UTC
- Manifest: `experiments/manifests/gsm8k_test_seed0_full.jsonl`
- SHA-256: `07f0f0fc10580dee941b6f921a3986854a8b0b74529d9bb952662d5daaea6bb2`
- Rows: `1,319`

The completed full-eval artefacts are:

- `eval/base_full.csv`
- `eval/r7_rloo_k8_step2000_full.csv`
- `eval/r7_rloo_k8_step3364_full.csv`
- `eval/r7_rloo_k8_best_full_ci.csv`
- `eval/r7_rloo_k8_best_full_ci.json`
- `eval/r7_rloo_k8_full_ci.csv`
- `eval/r7_rloo_k8_full_ci.json`
- `metadata/run_metadata.json`

The final checkpoint step `3364` was restored from
`/home/ext_harveybermingham1_gmail_com/tpu-runs/part-i/R7-rloo-k8-det-harvey-full-s0-20260611_105132/ckpts/actor`
and evaluated against the same full-test manifest as base and step `2000`.

## Results

| Model / checkpoint | Exact | Partial | Format | Empty |
|---|---:|---:|---:|---:|
| Base greedy | 625/1319 (47.38%) | 659/1319 (49.96%) | 53/1319 (4.02%) | 0/1319 |
| R7 RLOO K=8 step 2000 | 722/1319 (54.74%) | 764/1319 (57.92%) | 1242/1319 (94.16%) | 0/1319 |
| R7 RLOO K=8 step 3364 | 742/1319 (56.25%) | 770/1319 (58.38%) | 1203/1319 (91.21%) | 0/1319 |

Bootstrap confidence intervals used `10000` resamples and seed `12345`.

| Comparison | Delta exact | 95% CI | Result |
|---|---:|---:|---|
| R7 K=8 step 2000 vs base | +7.35 pp | [+4.62 pp, +10.16 pp] | CI excludes 0 |
| R7 K=8 step 3364 vs base | +8.87 pp | [+6.14 pp, +11.68 pp] | CI excludes 0 |

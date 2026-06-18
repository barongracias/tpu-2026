# H GRPO K=16 capped evidence

Run id: `H-grpo-k16-cap2500-s0-20260614-r2`

Run root:

`/home/harvey/tpu-runs/part-i/H-grpo-k16-cap2500-s0-20260614-r2`

## Full-test eval status

Completed against the shared full GSM8K test manifest:

- Manifest: `experiments/manifests/gsm8k_test_seed0_full.jsonl`
- SHA-256: `07f0f0fc10580dee941b6f921a3986854a8b0b74529d9bb952662d5daaea6bb2`
- Rows: `1,319`

Training stopped before the `2500` cap with JAX/XLA `RESOURCE_EXHAUSTED` after scalar step `678`. The last retained checkpoint is step `500`, so the full-test eval reports step `500` as both best retained and final / last-good.

The completed full-eval artefacts are:

- `eval/base_full.csv`
- `eval/H-grpo-k16-cap2500-s0-20260614-r2_step500_full.csv`
- `eval/H-grpo-k16-cap2500-s0-20260614-r2_full_ci.csv`
- `eval/H-grpo-k16-cap2500-s0-20260614-r2_full_ci.json`
- `metadata/run_metadata.json`

## Results

| Model / checkpoint | Exact | Partial | Format | Empty |
|---|---:|---:|---:|---:|
| Base greedy | 625/1319 (47.38%) | 659/1319 (49.96%) | 53/1319 (4.02%) | 0/1319 |
| GRPO K=16 step 500 | 703/1319 (53.30%) | 729/1319 (55.27%) | 1100/1319 (83.40%) | 0/1319 |

Bootstrap confidence intervals used `10000` resamples and seed `12345`.

| Comparison | Delta exact | 95% CI | Result |
|---|---:|---:|---|
| K=16 step 500 vs base | +5.91 pp | [+3.18 pp, +8.64 pp] | CI excludes 0 |

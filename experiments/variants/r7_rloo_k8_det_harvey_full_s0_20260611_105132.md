# R7 RLOO K=8 deterministic retained-checkpoint greedy eval summary

Run root: /home/ext_harveybermingham1_gmail_com/tpu-runs/part-i/R7-rloo-k8-det-harvey-full-s0-20260611_105132
Eval manifest: /home/ext_harveybermingham1_gmail_com/tpu-runs/part-i/manifests/gsm8k_test_seed0_n64.jsonl
Preset: greedy; source: tfds; prompts: 64; eval_seed: 0
Best retained checkpoint by exact accuracy: step 2000 (35/64, 54.69%)

| Step | Exact | Partial | Format | Empty |
| ---: | ---: | ---: | ---: | ---: |
| 250 | 28/64 (43.75%) | 29/64 (45.31%) | 49/64 (76.56%) | 0/64 |
| 500 | 30/64 (46.88%) | 31/64 (48.44%) | 53/64 (82.81%) | 0/64 |
| 750 | 33/64 (51.56%) | 34/64 (53.12%) | 55/64 (85.94%) | 0/64 |
| 1000 | 28/64 (43.75%) | 29/64 (45.31%) | 58/64 (90.62%) | 0/64 |
| 1250 | 27/64 (42.19%) | 29/64 (45.31%) | 61/64 (95.31%) | 0/64 |
| 1500 | 30/64 (46.88%) | 31/64 (48.44%) | 57/64 (89.06%) | 0/64 |
| 1750 | 30/64 (46.88%) | 30/64 (46.88%) | 59/64 (92.19%) | 0/64 |
| 2000 | 35/64 (54.69%) | 36/64 (56.25%) | 58/64 (90.62%) | 0/64 |
| 2250 | 28/64 (43.75%) | 30/64 (46.88%) | 57/64 (89.06%) | 0/64 |
| 2500 | 30/64 (46.88%) | 32/64 (50.00%) | 56/64 (87.50%) | 0/64 |
| 2750 | 31/64 (48.44%) | 32/64 (50.00%) | 53/64 (82.81%) | 0/64 |
| 3000 | 28/64 (43.75%) | 29/64 (45.31%) | 53/64 (82.81%) | 0/64 |
| 3250 | 27/64 (42.19%) | 28/64 (43.75%) | 54/64 (84.38%) | 0/64 |
| 3364 | 26/64 (40.62%) | 29/64 (45.31%) | 54/64 (84.38%) | 0/64 |

## Full-test confirmation - 2026-06-14 and 2026-06-15

Full GSM8K test manifest:

- `experiments/manifests/gsm8k_test_seed0_full.jsonl`
- SHA-256: `07f0f0fc10580dee941b6f921a3986854a8b0b74529d9bb952662d5daaea6bb2`
- Rows: `1,319`

Completed full evals:

| Model / checkpoint | Exact | Partial | Format | Empty |
|---|---:|---:|---:|---:|
| Base greedy | 625/1319 (47.38%) | 659/1319 (49.96%) | 53/1319 (4.02%) | 0/1319 |
| R7 RLOO K=8 step 2000 | 722/1319 (54.74%) | 764/1319 (57.92%) | 1242/1319 (94.16%) | 0/1319 |
| R7 RLOO K=8 step 3364 | 742/1319 (56.25%) | 770/1319 (58.38%) | 1203/1319 (91.21%) | 0/1319 |

Paired bootstrap over the 1,319 shared prompts used `10000` resamples and seed
`12345`.

| Comparison | Delta exact | 95% CI | Result |
|---|---:|---:|---|
| R7 K=8 step 2000 vs base | +7.35 pp | [+4.62 pp, +10.16 pp] | CI excludes 0 |
| R7 K=8 step 3364 vs base | +8.87 pp | [+6.14 pp, +11.68 pp] | CI excludes 0 |

The 64-prompt screen selected step `2000`, but the full-test final checkpoint
step `3364` is higher on exact accuracy. Report both checkpoint-selection facts:
step `2000` was the retained-screen winner, while step `3364` is the full-test
final checkpoint result.

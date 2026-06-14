# Evidence: Harvey full-test eval, 2026-06-14

## Scope

This note records the full GSM8K test-set confirmation eval for the deterministic Harvey RLOO K=2 run:

- Run id: `R7-rloo-k2-det-harvey-full-s0-20260611_102009`
- Commit: `71aab87dee2d2c78256384d084d063d8b40c9e0c`
- Method: RLOO
- K: `2`
- Seed: `0`
- Checkpoint policy: step `500` was selected as the best retained checkpoint from the existing n=64 sweep; step `3364` is the final checkpoint.

No R6 checkpoint was evaluated for this evidence note. R6 remains non-report-ready under the team register provenance criteria.

## Manifest

Full-test manifest:

- `experiments/manifests/gsm8k_test_seed0_full.jsonl`
- SHA-256: `07f0f0fc10580dee941b6f921a3986854a8b0b74529d9bb952662d5daaea6bb2`
- Rows: `1,319`

The manifest was built with `scripts/build_full_test_manifest.py --eval-seed 0` and verified against the existing n=64 manifest by `question_sha256` prefix. The n=64 manifest hash is:

- `9aa1814e295e155f4735c9302a7f9c28c6aaa2319074e053a5888e3f3b2448b0`

## Artefacts

Committed lightweight artefacts:

- `experiments/evidence/R7-rloo-k2-det-harvey-full-s0-20260611_102009/eval/base_full.csv`
- `experiments/evidence/R7-rloo-k2-det-harvey-full-s0-20260611_102009/eval/r7_rloo_k2_step500_full.csv`
- `experiments/evidence/R7-rloo-k2-det-harvey-full-s0-20260611_102009/eval/r7_rloo_k2_step3364_full.csv`
- `experiments/evidence/R7-rloo-k2-det-harvey-full-s0-20260611_102009/eval/r7_rloo_k2_full_ci.csv`
- `experiments/evidence/R7-rloo-k2-det-harvey-full-s0-20260611_102009/eval/r7_rloo_k2_full_ci.json`

Raw TPU-side logs remain under:

- `/home/harvey/tpu-runs/part-i/full_test_eval_20260614/logs/eval_base_full.log`
- `/home/harvey/tpu-runs/part-i/R7-rloo-k2-det-harvey-full-s0-20260611_102009/logs/eval_r7_rloo_k2_step500_full.log`
- `/home/harvey/tpu-runs/part-i/R7-rloo-k2-det-harvey-full-s0-20260611_102009/logs/eval_r7_rloo_k2_step3364_full.log`

## Results

| Model / checkpoint | Exact | Partial | Format | Empty |
|---|---:|---:|---:|---:|
| Base greedy | 625/1319 (47.38%) | 659/1319 (49.96%) | 53/1319 (4.02%) | 0/1319 |
| R7 RLOO K=2 step 500 | 562/1319 (42.61%) | 599/1319 (45.41%) | 1211/1319 (91.81%) | 0/1319 |
| R7 RLOO K=2 step 3364 | 59/1319 (4.47%) | 72/1319 (5.46%) | 161/1319 (12.21%) | 1111/1319 |

Bootstrap confidence intervals used `10000` resamples and seed `12345`.

| Comparison | Delta exact | 95% CI | Result |
|---|---:|---:|---|
| R7 step 500 vs base | -4.78 pp | [-7.66 pp, -1.82 pp] | CI excludes 0 |
| R7 step 3364 vs base | -42.91 pp | [-45.72 pp, -40.03 pp] | CI excludes 0 |
| R7 step 500 vs step 3364 | +38.13 pp | [+35.25 pp, +41.02 pp] | CI excludes 0 |

## Interpretation

The full-test eval confirms the n=64 diagnosis for deterministic RLOO K=2: the best retained checkpoint improves format compliance but remains below the base model on exact accuracy, while the final checkpoint collapses badly with many empty responses.

This evidence completes the same-manifest base comparison for Harvey R7 K=2. It does not complete the RLOO K=8 full-test comparison because the raw deterministic K=8 checkpoint tree was not present on this VM.

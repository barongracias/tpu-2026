# R7 K8 Full-Test Evaluation Log - 2026-06-14 and 2026-06-15

## Run Identity

- Run id: `R7-rloo-k8-det-harvey-full-s0-20260611_105132`
- Date/time started: 2026-06-14 UTC
- Final-checkpoint continuation: 2026-06-15 UTC
- Operator: Harvey VM account
- Purpose: confirm the selected R7 RLOO K=8 retained checkpoint and final checkpoint on the full GSM8K test split.
- Decision this run should enable: report-grade full-test accuracy and paired bootstrap CI for base vs R7 K=8 step `2000` and final step `3364`.

## Source State

- `tpu-2026` run commit: `e3d69a1938fe8e8a2a67a3c84a03331133ed46f1`
- Model revision: `dcc83ea841ab6100d6b47a070329e1ba4cf78752`
- Dependency refs: JAX `b6e3b4fecee2bc69fcf456c117f8854e20c06708`, Tunix `683256db1a0919b5cfd46cee52cebc96331494fb`, Qwix `5f33aade2fea630c3210a61a276c05b08d259949`, Flax `5ab9d1463550a7fc7fe13575af55081e68258dd7`.

## Configuration

- Data source: `tfds`
- Estimator: `rloo`
- Seed: `RUN_SEED=0`, `EVAL_SEED=0`
- Num generations: `8`
- Evaluation preset: `greedy`
- Checkpoint directory: `$HOME/tpu-runs/part-i/R7-rloo-k8-det-harvey-full-s0-20260611_105132/ckpts`
- Full eval manifest: `$HOME/tpu-2026/experiments/manifests/gsm8k_test_seed0_full.jsonl`
- Selected checkpoint: step `2000` best retained on the 64-prompt screen.
- Final checkpoint: step `3364`.

## Exact Commands

```bash
# The shared manifest already existed and was not rebuilt.
# Verified before launch:
# rows=1319
# sha256=07f0f0fc10580dee941b6f921a3986854a8b0b74529d9bb952662d5daaea6bb2

python -u evaluate.py --preset greedy --source tfds --no-restore   --eval-manifest "$EVAL_MANIFEST"   --output-csv "$RUN_ROOT/eval/base_full.csv"

python -u evaluate.py --preset greedy --source tfds   --ckpt-dir "$CKPT_DIR" --step 2000   --eval-manifest "$EVAL_MANIFEST"   --output-csv "$RUN_ROOT/eval/r7_rloo_k8_step2000_full.csv"

python -u evaluate.py --preset greedy --source tfds   --ckpt-dir "$CKPT_DIR" --step 3364   --eval-manifest "$EVAL_MANIFEST"   --output-csv "$RUN_ROOT/eval/r7_rloo_k8_step3364_full.csv"

python -u experiments/analysis/bootstrap_ci.py   --csv base="$RUN_ROOT/eval/base_full.csv"   --csv r7_k8_best="$RUN_ROOT/eval/r7_rloo_k8_step2000_full.csv"   --csv r7_k8_final="$RUN_ROOT/eval/r7_rloo_k8_step3364_full.csv"   --baseline base   --resamples 10000 --seed 12345   --output-json "$RUN_ROOT/eval/r7_rloo_k8_full_ci.json"   --output-csv "$RUN_ROOT/eval/r7_rloo_k8_full_ci.csv"
```

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

Output artefacts copied into the repo:

- `experiments/evidence/R7-rloo-k8-det-harvey-full-s0-20260611_105132/eval/base_full.csv`
- `experiments/evidence/R7-rloo-k8-det-harvey-full-s0-20260611_105132/eval/r7_rloo_k8_step2000_full.csv`
- `experiments/evidence/R7-rloo-k8-det-harvey-full-s0-20260611_105132/eval/r7_rloo_k8_step3364_full.csv`
- `experiments/evidence/R7-rloo-k8-det-harvey-full-s0-20260611_105132/eval/r7_rloo_k8_best_full_ci.csv`
- `experiments/evidence/R7-rloo-k8-det-harvey-full-s0-20260611_105132/eval/r7_rloo_k8_best_full_ci.json`
- `experiments/evidence/R7-rloo-k8-det-harvey-full-s0-20260611_105132/eval/r7_rloo_k8_full_ci.csv`
- `experiments/evidence/R7-rloo-k8-det-harvey-full-s0-20260611_105132/eval/r7_rloo_k8_full_ci.json`

Notes:

- Step `3364` was initially skipped on 2026-06-14 after an active TPU lock from another user's eval queue.
- The final full eval was completed on 2026-06-15 against the same manifest without rebuilding it.
- The n=64 retained-checkpoint screen selected step `2000`, but the full-test final checkpoint step `3364` is higher on exact accuracy.

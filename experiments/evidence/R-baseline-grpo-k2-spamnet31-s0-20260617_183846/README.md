# Baseline GRPO K=2 deterministic evidence

Run id: `R-baseline-grpo-k2-spamnet31-s0-20260617_183846`

Run root:

`/home/harvey/tpu-runs/part-i/R-baseline-grpo-k2-spamnet31-s0-20260617_183846`

## Full-test eval status

Completed against the shared full GSM8K test manifest:

- Manifest: `experiments/manifests/gsm8k_test_seed0_full.jsonl`
- SHA-256: `07f0f0fc10580dee941b6f921a3986854a8b0b74529d9bb952662d5daaea6bb2`
- Rows: `1,319`

The n=64 screening manifest was copied from the deterministic R7 K=2 evidence
protocol:

- Manifest: `manifests/gsm8k_test_seed0_n64.jsonl`
- SHA-256: `9aa1814e295e155f4735c9302a7f9c28c6aaa2319074e053a5888e3f3b2448b0`
- Rows: `64`

The run used the coursework default GRPO baseline settings:

- Model: `google/gemma-3-1b-it`
- Model revision: `dcc83ea841ab6100d6b47a070329e1ba4cf78752`
- Method: GRPO, `NUM_GENERATIONS=2`, seed `0`
- Commit: `7a77bce3a79783d5caa4ff782c1746012e4defe7`
- Dependency refs: JAX `3ff6dc40cdb982921de0ef12c0bf8e5c64311a6f`, Tunix `7fadb3c81e4348f714b2b5a07f6cc2bd10da10de`, Qwix `c2548f06eeca005090329530576623afcd48241c`, Flax `5ab9d1463550a7fc7fe13575af55081e68258dd7`

Checkpoint selection followed the deterministic manifest protocol: screen all
retained checkpoints on the n=64 manifest, select by exact-correct count, and
break ties by the lowest step. Steps `500` and `1250` both reached `30/64`; step
`500` was selected by the tie-break. Step `3364` is the final checkpoint.

The completed full-eval artefacts are:

- `eval/base_full.csv`
- `eval/R-baseline-grpo-k2-spamnet31-s0-20260617_183846_step500_full.csv`
- `eval/R-baseline-grpo-k2-spamnet31-s0-20260617_183846_step3364_full.csv`
- `eval/R-baseline-grpo-k2-spamnet31-s0-20260617_183846_full_ci.csv`
- `eval/R-baseline-grpo-k2-spamnet31-s0-20260617_183846_full_ci.json`
- `eval/R-baseline-grpo-k2-spamnet31-s0-20260617_183846_eval_validation.json`
- `eval/R-baseline-grpo-k2-spamnet31-s0-20260617_183846_n64_screen_metrics.csv`
- `eval/R-baseline-grpo-k2-spamnet31-s0-20260617_183846_n64_screen_summary.txt`
- `eval/R-baseline-grpo-k2-spamnet31-s0-20260617_183846_selected_steps.env`
- `metadata/run_metadata.json`
- `metadata/R-baseline-grpo-k2-spamnet31-s0-20260617_183846_eval_protocol.env`
- `metadata/launch_full_eval.sh`
- `metadata/run_eval_tmux.sh`

Raw TPU-side logs remain under:

- `/home/harvey/tpu-runs/part-i/R-baseline-grpo-k2-spamnet31-s0-20260617_183846/logs/train.log`
- `/home/harvey/tpu-runs/part-i/R-baseline-grpo-k2-spamnet31-s0-20260617_183846/logs/eval_pipeline.log`

## Results

| Model / checkpoint | Exact | Partial | Format | Empty |
|---|---:|---:|---:|---:|
| Base greedy | 625/1319 (47.38%) | 659/1319 (49.96%) | 53/1319 (4.02%) | 0/1319 |
| Baseline GRPO K=2 step 500 | 656/1319 (49.73%) | 688/1319 (52.16%) | 1038/1319 (78.70%) | 0/1319 |
| Baseline GRPO K=2 step 3364 | 218/1319 (16.53%) | 240/1319 (18.20%) | 469/1319 (35.56%) | 0/1319 |

Bootstrap confidence intervals used `10000` resamples and seed `12345`.

| Comparison | Delta exact | 95% CI | Result |
|---|---:|---:|---|
| Baseline step 500 vs base | +2.35 pp | [-0.30 pp, +5.00 pp] | CI includes 0 |
| Baseline step 3364 vs base | -30.86 pp | [-34.04 pp, -27.82 pp] | CI excludes 0 |

## Interpretation

The deterministic baseline reproduction confirms the earlier K=2 GRPO pattern:
the selected retained checkpoint is only modestly above the base point estimate
and its paired bootstrap interval includes zero, while the final checkpoint
collapses far below base. Format compliance rises sharply at the selected
checkpoint and then degrades alongside exact accuracy by the final checkpoint.

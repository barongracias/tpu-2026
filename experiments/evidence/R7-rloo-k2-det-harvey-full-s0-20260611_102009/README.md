# R7 RLOO K=2 deterministic evidence

Run id: `R7-rloo-k2-det-harvey-full-s0-20260611_102009`

This directory contains the lightweight CSV/JSON evidence copied from:

`/home/harvey/tpu-runs/part-i/R7-rloo-k2-det-harvey-full-s0-20260611_102009`

Contents:
- `eval/`: retained-checkpoint greedy per-prompt CSVs, metrics summary CSV, TensorBoard scalar CSV, and summary text files.
- `metadata/`: `run_metadata.json`, `launch_train.sh`, W&B metadata, and W&B summary.
- `manifests/gsm8k_test_seed0_n64.jsonl`: shared held-out eval manifest.

Full-test update, 2026-06-14:
- `eval/base_full.csv`: base model greedy eval on the full 1,319-row GSM8K test manifest.
- `eval/r7_rloo_k2_step500_full.csv`: best retained R7 K=2 checkpoint from the n=64 sweep, evaluated on the full test manifest.
- `eval/r7_rloo_k2_step3364_full.csv`: final R7 K=2 checkpoint, evaluated on the full test manifest.
- `eval/r7_rloo_k2_full_ci.csv` and `.json`: paired/bootstrap confidence intervals over the full-test CSVs.

Full manifest:
- `../../manifests/gsm8k_test_seed0_full.jsonl`
- SHA-256: `07f0f0fc10580dee941b6f921a3986854a8b0b74529d9bb952662d5daaea6bb2`

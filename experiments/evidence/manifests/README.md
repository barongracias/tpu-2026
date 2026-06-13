# Fred Hard-Medium Manifests 2026-06-12

## Files

- `gsm8k_train_base_hard_medium.jsonl`: 3,836 GSM8K train examples selected as hard or medium for the frozen base policy.
- `gsm8k_test_seed0_n64.jsonl`: 64-prompt held-out GSM8K test manifest for shared greedy evaluation with `EVAL_SEED=0`.

## Mining Protocol

- Source split: GSM8K train only.
- Mining script: `scripts/mine_hard_examples.py`.
- Probe model: frozen base `google/gemma-3-1b-it`.
- Model revision: `dcc83ea841ab6100d6b47a070329e1ba4cf78752`.
- Checkpoint restore: none; base policy only.
- Prompt template: `data.TEMPLATE+SYSTEM_PROMPT`.
- Greedy preset: `greedy`, one completion.
- Sample preset: `standard`, 8 completions.
- Sample seeds: `0,1,2,3,4,5,6,7`.
- Mining seed: `0`.
- Selection rule: include examples classified as hard or medium by base-policy probe results.

## Counts

- Total train examples processed: 7,473.
- Easy: 3,637.
- Medium: 1,990.
- Hard: 1,846.
- Hard plus medium manifest: 3,836.
- Failures/skipped: 0.

## Source Evidence

- Mining metadata: `../hard_mining_base_gsm8k_train_20260609_220117/run_metadata.json`.
- Mining summary: `../hard_mining_base_gsm8k_train_20260609_220117/summary.md`.

# Full Test-Set Evaluation Runbook (n = 1,319)

Purpose: produce report-grade accuracy numbers with tight uncertainty by
evaluating selected checkpoints on the **whole GSM8K test split (1,319 prompts)**
and computing bootstrap confidence intervals, instead of only the 64-prompt
screening set.

This runbook **extends** `experiments/runbooks/tpu_day1_runbook.md` §8
("Evaluation Restore Check") and obeys `experiments/manifests/experiment_contract.md`.
It does **not** replace them. The 64-prompt manifest stays the cheap *screening*
set used to pick checkpoints; this adds a *confirmation* eval for the I.3
uncertainty requirement in `experiments/manifests/i3_sweep_plan.md`. Nothing here
changes any baseline-owned file (`config.py`, `data.py`, `train.py`,
`evaluate.py`, `bootstrap.sh`); it only adds two helper scripts and a shared
manifest.

## Why this is needed

- `config.py` caps the held-out set at `NUM_TEST_BATCHES = 64` *only when no
  manifest is supplied*. When `evaluate.py` is given an existing `--eval-manifest`
  JSONL it loads **every row in that file** and ignores the cap. So "evaluate on
  the whole test set" means "supply a manifest that contains the whole test set".
- At n = 64 a 50%-accuracy point has a 95% CI of roughly ±12 points, so the
  K-sweep differences cannot be resolved. At n = 1,319 the CI shrinks to roughly
  ±3 points.

## Ground rules (carried over, do not relax)

- Do not start any eval until the debug/full-run gates in the day-one runbook pass.
- Keep all artefacts outside `/tmp`; use the run root.
- Record the eval in an iteration log before launching (template in
  `experiments/templates/`).
- Do not push from the VM. Commit lightweight evidence only (see step 6).
- Do not write tooling/assistant references into any committed file.

## Prerequisites

Set the same pins, seeds, and paths the contract already requires:

```bash
source ~/venvs/tunix/bin/activate
cd ~/tpu-2026/scripts

export RUN_SEED=0
export EVAL_SEED=0          # MUST match the seed that produced gsm8k_test_seed0_n64.jsonl
# MODEL_REVISION / JAX_REF / QWIX_REF / FLAX_REF / TUNIX_REF as per experiment_contract.md
# RUN_ROOT / CKPT_DIR / TEST_DATA_DIR as per experiment_contract.md, e.g.:
export RUN_ROOT="$HOME/tpu-runs/part-i/<run-id>"
```

## Step 1 — Build the shared full-test manifest (once for the whole team)

Run on a dataset-capable VM (needs TFDS). Build it into the repo so it is
version-controlled and shareable by hash; this satisfies the contract's
"prefer a shared manifest path outside any single run root".

```bash
python -u build_full_test_manifest.py \
  --out ~/tpu-2026/experiments/manifests/gsm8k_test_seed0_full.jsonl \
  --eval-seed 0 \
  --verify-prefix ~/tpu-2026/experiments/evidence/R7-rloo-k2-det-harvey-full-s0-20260611_102009/manifests/gsm8k_test_seed0_n64.jsonl
```

Expected:
- `PREFIX OK: first 64 rows ... match ... by question_sha256` — confirms the
  existing n=64 set is exactly the first 64 rows of the full manifest, so
  `prompt_id` alignment holds between screening and full evals.
- A printed `SHA-256` and a `.sha256` sidecar. **Record that hash and share it**;
  it is the full-test analogue of the n=64 hash
  `9aa1814e295e155f4735c9302a7f9c28c6aaa2319074e053a5888e3f3b2448b0`.

Everyone (Baron, Fred, Harvey) must evaluate against the **byte-identical** file.
Point `EVAL_MANIFEST` at the committed copy (or copy it to
`$HOME/tpu-runs/part-i/manifests/` first — just keep the SHA identical):

```bash
export EVAL_MANIFEST=~/tpu-2026/experiments/manifests/gsm8k_test_seed0_full.jsonl
```

## Step 2 — Select checkpoints on the 64-prompt screen (do NOT full-eval all)

Full-test eval is greedy, one prompt at a time, so it costs ~1,319 generations
per checkpoint (~20× a 64-prompt eval). Do not run it on every retained
checkpoint. Use the existing per-checkpoint 64-prompt CSVs to choose, per run:

- the **best retained** checkpoint (top 1–2 candidates by 64-prompt exact), and
- the **final** checkpoint.

Plus the **base model** once. A typical selection is base + ~2 checkpoints per run.

## Step 3 — Base model on the full set

This is the anchor for every comparison and is currently missing on the same
manifest. Run it once per manifest:

```bash
python -u evaluate.py --preset greedy --source tfds --no-restore \
  --eval-manifest "$EVAL_MANIFEST" \
  --output-csv "$RUN_ROOT/eval/base_full.csv"
```

## Step 4 — Selected checkpoints on the full set

For each selected step (repeat per run; example shows K=8 best then final):

```bash
python -u evaluate.py --preset greedy --source tfds \
  --ckpt-dir "$RUN_ROOT/ckpts" --step 2000 \
  --eval-manifest "$EVAL_MANIFEST" \
  --output-csv "$RUN_ROOT/eval/<run-id>_step2000_full.csv"

python -u evaluate.py --preset greedy --source tfds \
  --ckpt-dir "$RUN_ROOT/ckpts" --step 3364 \
  --eval-manifest "$EVAL_MANIFEST" \
  --output-csv "$RUN_ROOT/eval/<run-id>_step3364_full.csv"
```

Gate (same spirit as day-one runbook §8):
- Each CSV has 1,319 rows and a populated `restored_step` matching the requested
  step (base CSV has `restored_step` empty / `--no-restore`).
- Every CSV records the same `eval_manifest` path.

## Step 5 — Bootstrap confidence intervals

Run the shared analysis (pure stdlib; runs on the VM or a laptop). Use the base
as the paired baseline so each model is compared on the same prompts:

```bash
python -u ../experiments/analysis/bootstrap_ci.py \
  --csv base=$RUN_ROOT/eval/base_full.csv \
  --csv k8_best=$RUN_ROOT/eval/<run-id>_step2000_full.csv \
  --csv k8_final=$RUN_ROOT/eval/<run-id>_step3364_full.csv \
  --baseline base \
  --resamples 10000 --seed 12345 \
  --output-json $RUN_ROOT/eval/<run-id>_full_ci.json \
  --output-csv  $RUN_ROOT/eval/<run-id>_full_ci.csv
```

This emits per-model accuracy with 95% CIs and paired difference CIs
(`*` flags a 95% CI that excludes 0). Everyone must use the **same** `--seed`
and `--resamples` so the uncertainty column is comparable across runs.

## Step 6 — What to commit

Commit lightweight evidence only; do not push:
- `experiments/manifests/gsm8k_test_seed0_full.jsonl` and its `.sha256`,
- the per-prompt `*_full.csv` files (base + selected checkpoints),
- the `*_full_ci.json` / `*_full_ci.csv` outputs,
- a short summary in the run's evidence/variant note.

Do **not** commit checkpoints, raw `events.out.*` files, `.env`, or full
`$HOME/tpu-runs` trees, and do not add tooling/assistant references.

## Caveats to disclose in the report

- The 64 screening prompts are a subset of the 1,319, and they influenced
  checkpoint selection, so a "best retained" full-test score is mildly optimistic.
  Disclose this, or (extra-clean) select on the 64 and report on the remaining
  1,255. Overlap is ~5%, so disclosure is normally sufficient.
- A K=16 run capped before full steps (e.g. step 2,500) is not directly
  compute-matched to full K=2/4/8 runs; report it to its last good step and label
  it as such, consistent with how the register treats capped/OOM runs.
- Bootstrap CI over eval prompts is the **primary** uncertainty; any second-seed
  spread is a secondary sanity check only.
```

# H GRPO K=4 Full Seed 0 Result 2026-06-14

## Run Identity

- Run id: `H-grpo-k4-full-s0-20260614`
- Date/time started: `2026-06-14T15:48:00Z`
- Operator: Harvey
- Machine: Harvey TPU VM, host `t1v-n-6e5e72ce-w-0`
- Purpose: add the GRPO `K=4` point to the group-size sweep.
- Hypothesis: increasing GRPO group size from `K=2` toward `K=4` should reduce estimator variance relative to the smallest-group baseline without the memory cost of `K=16`.
- Decision this run should enable: select the K=4 best retained checkpoint on the 64-prompt screen, then confirm base/best/final performance on the full 1,319-prompt GSM8K test manifest.

## Source State

- `tpu-2026` branch: `harvey-ksweep`
- `tpu-2026` commit: `57c6409add0d81bbdb32ca7f4b3e176b4e044068`
- `jax` commit: `3ff6dc40cdb982921de0ef12c0bf8e5c64311a6f`
- `tunix` commit: `7fadb3c81e4348f714b2b5a07f6cc2bd10da10de`
- `qwix` commit: `c2548f06eeca005090329530576623afcd48241c`
- `flax` commit: `5ab9d1463550a7fc7fe13575af55081e68258dd7`
- Local patches applied: none to training or evaluation code for this run; `NUM_GENERATIONS` is an environment knob.
- Dirty files intentionally present before evaluation: full-test eval runbook/helper and full manifest evidence files staged in the working tree for this follow-up evaluation.

## Configuration

- Data source: `tfds`
- Estimator: `grpo`
- Seed: `RUN_SEED=0`; `EVAL_SEED=0`
- Max steps: `3364`
- Num generations: `4`
- Num iterations: `1`
- Beta: `0.08`
- Epsilon: `0.2`
- Batch size / rollout settings: `TRAIN_MICRO_BATCH_SIZE=1`, greedy eval preset, `TOTAL_GENERATION_STEPS=768`
- Screening eval manifest: `/home/harvey/tpu-runs/part-i/manifests/gsm8k_test_seed0_n64.jsonl`, SHA-256 `9aa1814e295e155f4735c9302a7f9c28c6aaa2319074e053a5888e3f3b2448b0`
- Full eval manifest: `experiments/manifests/gsm8k_test_seed0_full.jsonl`, SHA-256 `07f0f0fc10580dee941b6f921a3986854a8b0b74529d9bb952662d5daaea6bb2`
- Checkpoint directory: `/home/harvey/tpu-runs/part-i/H-grpo-k4-full-s0-20260614/ckpts`
- Intermediate checkpoint directory: `/home/harvey/tpu-runs/part-i/H-grpo-k4-full-s0-20260614/intermediate_ckpt`
- TensorBoard directory: `/home/harvey/tpu-runs/part-i/H-grpo-k4-full-s0-20260614/tensorboard`
- W&B entity/project/run id: `barongracias-university-of-cambridge` / `agentic-ai-coursework` / `H-grpo-k4-full-s0-20260614`

## Exact Commands

Training command:

```bash
ADV_ESTIMATOR=grpo NUM_GENERATIONS=4 RUN_SEED=0 EVAL_SEED=0 \
MODEL_REVISION=dcc83ea841ab6100d6b47a070329e1ba4cf78752 \
SAVE_INTERVAL_STEPS=250 MAX_TO_KEEP=20 \
EVAL_MANIFEST=/home/harvey/tpu-2026/experiments/manifests/gsm8k_test_seed0_full.jsonl \
RUN_ROOT=/home/harvey/tpu-runs/part-i/H-grpo-k4-full-s0-20260614 \
WANDB_ENTITY=barongracias-university-of-cambridge \
WANDB_PROJECT=agentic-ai-coursework \
WANDB_RUN_ID=H-grpo-k4-full-s0-20260614 \
./scripts/run_tmux.sh
```

Evaluation command:

```bash
tmux new-session -d -s eval-H-grpo-k4-full-s0-20260614 \
  'bash /home/harvey/tpu-runs/part-i/eval_k4_runbook_20260614.sh'
```

Plot/extraction command:

```bash
python -u experiments/analysis/bootstrap_ci.py \
  --csv base=/home/harvey/tpu-runs/part-i/H-grpo-k4-full-s0-20260614/eval/base_full.csv \
  --csv k4_best=/home/harvey/tpu-runs/part-i/H-grpo-k4-full-s0-20260614/eval/H-grpo-k4-full-s0-20260614_step<BEST>_full.csv \
  --csv k4_final=/home/harvey/tpu-runs/part-i/H-grpo-k4-full-s0-20260614/eval/H-grpo-k4-full-s0-20260614_step3364_full.csv \
  --baseline base \
  --resamples 10000 --seed 12345 \
  --output-json /home/harvey/tpu-runs/part-i/H-grpo-k4-full-s0-20260614/eval/H-grpo-k4-full-s0-20260614_full_ci.json \
  --output-csv /home/harvey/tpu-runs/part-i/H-grpo-k4-full-s0-20260614/eval/H-grpo-k4-full-s0-20260614_full_ci.csv
```

## Live Checks

- TPU backend confirmed: yes, one TPU device visible before launch.
- W&B run visible: yes, `https://wandb.ai/barongracias-university-of-cambridge/agentic-ai-coursework/runs/H-grpo-k4-full-s0-20260614`
- TensorBoard events written: yes.
- Checkpoints written: yes, `1`, every 250 steps through `3250`, and final `3364`.
- Estimator logged correctly: yes, `grpo`.
- Reward metric present: yes, scalar export copied under `experiments/evidence/H-grpo-k4-full-s0-20260614/scalars/`.
- KL metric present: yes, scalar export copied under `experiments/evidence/H-grpo-k4-full-s0-20260614/scalars/`.
- Diagnostic metrics present: yes, scalar export copied under `experiments/evidence/H-grpo-k4-full-s0-20260614/scalars/`.
- Any warnings or crashes: W&B step-order warnings at training end; no fatal/OOM/traceback found before evaluation.

## Results

- Training completed: yes.
- Final step: `3364`.
- Wall-clock time: not extracted for this note.
- Final/representative reward: see scalar CSV in the evidence directory.
- Final/representative KL: see scalar CSV in the evidence directory.
- Evaluation sample count: screening `64`; full confirmation `1,319`.
- Evaluation accuracy or score:
  - Base greedy: `625/1319` exact, `47.38%`.
  - K=4 best retained step `2750`: `682/1319` exact, `51.71%`.
  - K=4 final step `3364`: `681/1319` exact, `51.63%`.
- Confidence interval:
  - K=4 step `2750` vs base: `+4.32 pp`, 95% CI `[+1.36 pp, +7.20 pp]`.
  - K=4 step `3364` vs base: `+4.25 pp`, 95% CI `[+1.67 pp, +6.90 pp]`.
- Saved per-prompt output: yes, under `experiments/evidence/H-grpo-k4-full-s0-20260614/eval/`.
- Saved plots: not applicable.
- Saved logs: raw TPU-side logs remain under `/home/harvey/tpu-runs/part-i/H-grpo-k4-full-s0-20260614/logs/`.

## Interpretation

- What happened: GRPO K=4 completed the full training budget and both selected checkpoints beat the base model on the shared full GSM8K test manifest.
- Whether the result supports the hypothesis: yes, K=4 improves exact accuracy over base with paired bootstrap intervals excluding zero.
- Algorithm-effectiveness interpretation: K=4 is a positive group-size sweep point, but it is weaker than the later K=16 last-good checkpoint and the RLOO K=8 final result recorded elsewhere.
- Edge cases or instability observed: no fatal training crash; W&B step-order warnings at the end were non-fatal.
- Alternative explanations: best retained step `2750` was chosen on the 64-prompt screen, so its full-test result carries the runbook's checkpoint-selection caveat.
- What changed relative to previous iteration: this is the first full K=4 GRPO point on Harvey's VM for the group-size sweep.

## Decision

- Keep / rerun / discard: keep.
- Reason: K=4 gives a statistically positive full-test result against base and completed cleanly.
- Next run: K=16 capped run was launched and later evaluated to its last-good checkpoint after early memory exhaustion.
- Report relevance: group-size sweep evidence for the `K_eff = K - 1` variance argument.
- Evidence path(s) to cite:
  - `experiments/evidence/H-grpo-k4-full-s0-20260614/`
  - `experiments/team_evidence/evidence_harvey_k4_full_eval_20260614.md`
- Caveats for report wording: best retained checkpoint is selected on the 64-prompt screen before full-test confirmation, so disclose mild selection optimism.

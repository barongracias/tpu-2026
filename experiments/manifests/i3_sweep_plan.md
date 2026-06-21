# I.1 / I.3 run plan (LOCK before the TPU window opens 2026-06-08)

Supersession note (2026-06-09): this file records the pre-TPU plan. The actual run sequence pivoted after R1/R3 diagnostics: R1 GRPO K=2 and R3 RLOO K=2 exposed reward/control failures, D3/D4 tested GRPO K=8, and R5 became the full GRPO K=8 seed-0 run. Current results and decisions live in `agents/report_notes.md` and the per-run notes under `experiments/`.

Status: SUPERSEDED DRAFT. Original status was draft to lock with the team. Team-shared v6e-1 TPU, window Mon 8 - Mon 15 Jun 2026.
Baseline: `tpu-2026` @ `324abbe`; Tunix git HEAD (`external/tunix` @ `683256d`), installed via `bootstrap.sh` (NOT PyPI `google-tunix`).

## Objective

- I.1 (10 marks): reproduce the baseline GRPO finetune of `google/gemma-3-1b-it` on GSM8K.
- I.3 (30 marks): one principled, controlled modification + comparison; >= 3 complete runs total; table with seeds + uncertainty; reward/KL curves on shared axes; one diagnostic plot; discussion tied to lecture theory.

## The I.3 modification (chosen): advantage estimator `grpo` -> `rloo`

What: switch the group-relative advantage from the STANDARD group-mean+std normalisation
(`compute_advantages`, algo_core.py:549) to LEAVE-ONE-OUT (`compute_rloo_advantages`,
algo_core.py:570), selected by Tunix's `GRPOConfig.advantage_estimator` field
(default `"grpo"`; alternatives `"rloo"`, `"drgrpo"`). Config-only; no algorithm code written.

Why (theory-grounded, ties to I.4 Q1):
- At the baseline's K = `NUM_GENERATIONS` = 2, standard GRPO gives `A_hat_i = (r_i - r_bar)/(sigma_r + 1e-6)`
  = `+/- 1/sqrt(2)` — only the SIGN of which sibling is better survives (magnitude normalised away),
  and the group is degenerate (sigma_r -> 0, ~0 advantage) whenever the two rewards are equal.
- RLOO at K=2 gives `A_hat_i = r_i - r_j` — keeps the reward-difference MAGNITUDE and degrades
  gracefully as rewards converge (no 1/sigma_r blow-up, no degenerate-group dead zone).
- So the swap is a direct empirical test of the group-mean-baseline pathology derived in I.4 Q1
  (the -1/K advantage correlation and the sigma_r normalisation), at the small K the baseline uses.

Why this modification (vs alternatives): it is COMPUTE-MATCHED by construction (same K, same
generations, same steps -> same wall-clock), so the I.3 "same total compute" control is automatic;
it is a one-field change (low TPU-week risk); and it speaks to the lecturer's efficiency framing
(better signal per rollout at fixed compute). Group-size sweeps (K=2->4->8) are compute-confounded
and are a stretch only.

## Run matrix

Target 4 runs (2 seeds x 2 estimators) for clean uncertainty; FLOOR is 3 (drop R2 if time is lost).
~5 h/run (lecturer) => ~20 h core of the shared week, leaving margin for debug/eval/failures.

| ID | advantage_estimator | seed | role | priority |
|----|---------------------|------|------|----------|
| R1 | grpo (standard)     | s0   | I.1 reproduction + I.3 control | MUST |
| R3 | rloo (leave-one-out)| s0   | I.3 variant | MUST |
| R4 | rloo                | s1   | variant 2nd seed (uncertainty) | MUST (=3rd run) |
| R2 | grpo                | s1   | baseline 2nd seed (uncertainty) | if time |
| R5 | drgrpo              | s0   | 3rd estimator (no std norm) | stretch |

(Run order R1, R3, R4 first so the floor of 3 is met early; then R2, R5 if the window allows.)

## Controlled-comparison invariants (HOLD FIXED across every run)

Only `advantage_estimator` (and the training seed for seed-repeats) may change. Everything else fixed:
- model `google/gemma-3-1b-it`; data + train/val/test split; `TRAIN_FRACTION=0.9`; `NUM_BATCHES=3738`;
  `MAX_STEPS=int(NUM_BATCHES*NUM_ITERATIONS*TRAIN_FRACTION*NUM_EPOCHS)=3364` (same total optimisation steps = same compute);
- `NUM_GENERATIONS`=K=2; `NUM_ITERATIONS`=1; sampling `TEMPERATURE=0.9`, `TOP_P=1.0`, `TOP_K=50`;
  `MAX_PROMPT_LENGTH=256`, `TOTAL_GENERATION_STEPS=768`;
- `BETA`=0.08, `EPSILON`=0.2; LoRA `RANK`=64, `ALPHA`=64; `LEARNING_RATE`=3e-6, adamw b1=0.9/b2=0.99/wd=0.1,
  warmup-cosine, `MAX_GRAD_NORM`=0.1;
- eval: same held-out GSM8K test split + same eval seed for base model and every checkpoint.

## Baseline patch needed (document in the report's "baseline patches" section)

1. Thread `advantage_estimator` through: `train.py` currently builds
   `GRPOConfig(num_generations=, num_iterations=, beta=, epsilon=)` (train.py:161-166) and so defaults
   to `"grpo"`. Add `advantage_estimator=ADV_ESTIMATOR` (new `config.py` knob, env/CLI-settable).
   VERIFY on the debug run that `"rloo"` actually routes to `compute_rloo_advantages`.
2. Lock seeds: set `GRPOLearner(..., data_shuffle_seed=SEED)` and the rollout/JAX PRNG seed; record
   the exact seed plumbing once confirmed on the debug run (sampling is stochastic at TEMPERATURE=0.9).

## Eval protocol + uncertainty

- Run `scripts/evaluate.py` on (i) the base model and (ii) each finetuned LoRA checkpoint, on the SAME
  held-out GSM8K split with a FIXED eval seed. Metric: GSM8K accuracy (boxed-answer exact match).
- Uncertainty: PRIMARY = bootstrap CI over the eval prompts (resample the held-out set; one eval run per
  checkpoint, cheap). SECONDARY = between-seed spread from the s0/s1 repeats.
- Report a table: base model | grpo ckpt (s0[, s1]) | rloo ckpt (s0, s1)[ | drgrpo] with mean +/- CI and seeds.

## Diagnostic plot (I.3 requires one GRPO failure mode)

Fraction of DEGENERATE groups (all K rewards equal => sigma_r=0 => ~0 advantage) over training, grpo vs rloo.
Cheap source: Tunix logs `advantage/nonzero_frac` (and `pg_clipfrac`, `is_ratio` spread, `kl`) in the loss
aux (algo_core.py ~493-510) -> degenerate fraction ~= 1 - nonzero_frac, no extra instrumentation.
Backup options: distribution of A_hat_i over training; response length / entropy; KL(pi_theta||pi_ref) vs accuracy.

## Pre-flight checklist (DAY 1, before spending any 5 h run)

1. Env: install Tunix from git HEAD per `bootstrap.sh` (NOT PyPI). Confirm
   `from tunix.rl.grpo.grpo_learner import GRPOConfig, GRPOLearner` imports and `GRPOConfig` accepts `advantage_estimator`.
2. Access: connect via VSCode SSH; run everything under TMUX (long runs survive disconnects).
3. CHECKPOINTS OFF /tmp: `config.py:67-69` points `CKPT_DIR`/`INTERMEDIATE_CKPT_DIR`/`TENSORBOARD_DIR` at
   `/tmp/content/...` and warns /tmp is VOLATILE. Repoint to persistent storage before any long run, or lose checkpoints.
4. W&B: `config.py` defaults to the shared team project and entity. Set `WANDB_API_KEY`, override
   `WANDB_PROJECT`/`WANDB_ENTITY` only deliberately, and confirm reward/KL/aux metrics log.
5. Secrets: `HF_TOKEN` for the gemma-3-1b-it download; confirm model + GSM8K (`DATA_SOURCE`) load.
6. Debug run: `MAX_STEPS ~= 50`, end-to-end (train -> checkpoint -> `evaluate.py` on base + ckpt -> accuracy ->
   W&B). Time it to extrapolate the full-run wall-clock and confirm it fits ~5 h.
7. Test the patch on debug: run BOTH `grpo` and `rloo` for a few steps; confirm the estimator switch takes effect.
8. Lock + record: seeds (data + sampling), eval split + eval seed, exact tunix commit, config diff per run.

## Outputs to capture per run (for I.1 / I.3 + viva)

- exact tunix commit, exact `tpu-2026` commit, full config, advantage_estimator, seed(s);
- wall-clock + GRPO steps actually completed; W&B/TensorBoard logs (mean reward, KL, the aux diagnostics);
- Orbax checkpoint path (persistent); base + checkpoint GSM8K accuracy on the fixed eval split;
- any patch/fix applied (for the "baseline patches" section).

## Fallback if TPU time is lost (per the brief)

Floor = 3 complete runs (R1, R3, R4). If fewer complete, keep R1 (baseline) + R3 (rloo s0) as the minimal
controlled pair and document the lost time + reason in "baseline patches"; lean on bootstrap-CI uncertainty
from the runs that did finish. Prioritise completing whole runs over starting many partial ones.

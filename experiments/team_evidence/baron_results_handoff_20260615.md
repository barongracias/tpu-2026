# Harvey Baron Package - 2026-06-15

Purpose: answer Baron's requested Harvey handoff for the GRPO K sweep and already-available RLOO full-test evals. Paths are relative to the `tpu-2026` repo unless stated otherwise.

Evidence collation branch at handoff time: `harvey-grpo-k8-rerun`, commit `742de13a7a220c9bb46aa7ad982a17e8fbbc77c7`.

## Headline Status

| Run | Status | Best retained / last-good | Final checkpoint | Full-test exact on manifest `07f0f0fc...` |
|---|---|---:|---:|---:|
| `H-grpo-k4-full-s0-20260614` | Completed full `3364`-step GRPO run. Not capped. No OOM. | step `2750`, `682/1319` | step `3364`, `681/1319` | best `51.71%`; final `51.63%` |
| `H-grpo-k16-cap2500-s0-20260614-r2` | Ran with `MAX_STEPS_OVERRIDE=2500`, then stopped early with JAX/XLA `RESOURCE_EXHAUSTED` after scalar step `678`. | step `500`, `703/1319` | step `500` last-good, `703/1319` | `53.30%` |

K=4 is a completed positive GRPO group-size point. K=16 is positive at the last-good checkpoint, but it must be reported as capped/early-stopped/OOM rather than a completed or compute-matched run.

## Shared Manifest

- Full-test manifest: `experiments/manifests/gsm8k_test_seed0_full.jsonl`
- SHA-256: `07f0f0fc10580dee941b6f921a3986854a8b0b74529d9bb952662d5daaea6bb2`
- Rows: `1,319`
- n=64 screening manifest SHA-256: `9aa1814e295e155f4735c9302a7f9c28c6aaa2319074e053a5888e3f3b2448b0`

## GRPO K=4 Package

- Run id: `H-grpo-k4-full-s0-20260614`
- Launch branch: `harvey-ksweep`
- Launch commit: `57c6409add0d81bbdb32ca7f4b3e176b4e044068`
- Method / K / seed: GRPO, `K=4`, seed `0`
- W&B URL: `https://wandb.ai/barongracias-university-of-cambridge/agentic-ai-coursework/runs/H-grpo-k4-full-s0-20260614`
- Runtime: not extracted into the committed local artefacts; the committed result note records the W&B URL but says wall-clock was not extracted.
- Config: `RUN_SEED=0`, `EVAL_SEED=0`, `MAX_STEPS=3364`, `NUM_GENERATIONS=4`, `NUM_ITERATIONS=1`, `beta=0.08`, `epsilon=0.2`, `SAVE_INTERVAL_STEPS=250`, `MAX_TO_KEEP=20`, `TOTAL_GENERATION_STEPS=768`, `MODEL_REVISION=dcc83ea841ab6100d6b47a070329e1ba4cf78752`.
- Result note: `experiments/variants/h_grpo_k4_full_s0_20260614.md`
- Full-test evidence note: `experiments/team_evidence/evidence_harvey_k4_full_eval_20260614.md`
- Evidence directory: `experiments/evidence/H-grpo-k4-full-s0-20260614/`
- Metadata: `experiments/evidence/H-grpo-k4-full-s0-20260614/metadata/run_metadata.json`
- Scalar CSV: `experiments/evidence/H-grpo-k4-full-s0-20260614/scalars/H-grpo-k4-full-s0-20260614_scalars.csv`
- Scalar coverage: reward tags under `rewards/train/*` and `rewards/eval/*`, KL tags under `actor/train/kl` and `actor/eval/kl`, and completion-length tags under `completions/train/*` and `completions/eval/*`.
- n=64 summary: `experiments/evidence/H-grpo-k4-full-s0-20260614/eval/H-grpo-k4-full-s0-20260614_n64_screen_summary.txt`
- n=64 metrics CSV: `experiments/evidence/H-grpo-k4-full-s0-20260614/eval/H-grpo-k4-full-s0-20260614_n64_screen_metrics.csv`
- n=64 per-checkpoint CSVs: `experiments/evidence/H-grpo-k4-full-s0-20260614/eval/H-grpo-k4-full-s0-20260614_step{250,500,750,1000,1250,1500,1750,2000,2250,2500,2750,3000,3250,3364}_greedy.csv`
- Full-test CSVs: `experiments/evidence/H-grpo-k4-full-s0-20260614/eval/base_full.csv`, `experiments/evidence/H-grpo-k4-full-s0-20260614/eval/H-grpo-k4-full-s0-20260614_step2750_full.csv`, `experiments/evidence/H-grpo-k4-full-s0-20260614/eval/H-grpo-k4-full-s0-20260614_step3364_full.csv`
- Full-test CI files: `experiments/evidence/H-grpo-k4-full-s0-20260614/eval/H-grpo-k4-full-s0-20260614_full_ci.csv`, `experiments/evidence/H-grpo-k4-full-s0-20260614/eval/H-grpo-k4-full-s0-20260614_full_ci.json`

K=4 n=64 screen: best retained checkpoint step `2750` with `37/64` exact; final step `3364` was `28/64`.

K=4 full-test result:

| Model / checkpoint | Exact | Partial | Format | Empty |
|---|---:|---:|---:|---:|
| Base greedy | `625/1319` (`47.38%`) | `659/1319` (`49.96%`) | `53/1319` (`4.02%`) | `0/1319` |
| K=4 step `2750` | `682/1319` (`51.71%`) | `712/1319` (`53.98%`) | `1104/1319` (`83.70%`) | `0/1319` |
| K=4 step `3364` | `681/1319` (`51.63%`) | `712/1319` (`53.98%`) | `1097/1319` (`83.17%`) | `0/1319` |

Paired bootstrap exact deltas vs base, `10000` resamples, seed `12345`: step `2750` is `+4.32 pp` with 95% CI `[+1.36, +7.20]`; step `3364` is `+4.25 pp` with 95% CI `[+1.67, +6.90]`.

## GRPO K=16 Package

- Run id: `H-grpo-k16-cap2500-s0-20260614-r2`
- Launch branch: `harvey-ksweep`
- Launch commit: `57c6409add0d81bbdb32ca7f4b3e176b4e044068`
- Method / K / seed: GRPO, `K=16`, seed `0`
- W&B URL: `https://wandb.ai/barongracias-university-of-cambridge/agentic-ai-coursework/runs/H-grpo-k16-cap2500-s0-20260614-r2`
- Runtime: not extracted into the committed local artefacts; local evidence records the stop condition after scalar step `678`.
- Config: `RUN_SEED=0`, `EVAL_SEED=0`, `MAX_STEPS_OVERRIDE=2500`, `lr_decay_steps=3364`, `NUM_GENERATIONS=16`, `NUM_ITERATIONS=1`, `beta=0.08`, `epsilon=0.2`, `SAVE_INTERVAL_STEPS=250`, `MAX_TO_KEEP=20`, `TOTAL_GENERATION_STEPS=768`, `MODEL_REVISION=dcc83ea841ab6100d6b47a070329e1ba4cf78752`.
- Result note: `experiments/variants/h_grpo_k16_cap2500_s0_20260614_r2.md`
- Full-test evidence note: `experiments/team_evidence/evidence_harvey_k16_full_eval_20260615.md`
- Evidence directory: `experiments/evidence/H-grpo-k16-cap2500-s0-20260614-r2/`
- Metadata: `experiments/evidence/H-grpo-k16-cap2500-s0-20260614-r2/metadata/run_metadata.json`
- Scalar CSV: `experiments/evidence/H-grpo-k16-cap2500-s0-20260614-r2/scalars/H-grpo-k16-cap2500-s0-20260614-r2_scalars.csv`
- Scalar coverage: reward tags under `rewards/train/*` and `rewards/eval/*`, KL tags under `actor/train/kl` and `actor/eval/kl`, and completion-length tags under `completions/train/*` and `completions/eval/*`.
- n=64 summary: `experiments/evidence/H-grpo-k16-cap2500-s0-20260614-r2/eval/H-grpo-k16-cap2500-s0-20260614-r2_n64_screen_summary.txt`
- n=64 metrics CSV: `experiments/evidence/H-grpo-k16-cap2500-s0-20260614-r2/eval/H-grpo-k16-cap2500-s0-20260614-r2_n64_screen_metrics.csv`
- n=64 per-checkpoint CSVs: `experiments/evidence/H-grpo-k16-cap2500-s0-20260614-r2/eval/H-grpo-k16-cap2500-s0-20260614-r2_step250_greedy.csv`, `experiments/evidence/H-grpo-k16-cap2500-s0-20260614-r2/eval/H-grpo-k16-cap2500-s0-20260614-r2_step500_greedy.csv`
- Full-test CSVs: `experiments/evidence/H-grpo-k16-cap2500-s0-20260614-r2/eval/base_full.csv`, `experiments/evidence/H-grpo-k16-cap2500-s0-20260614-r2/eval/H-grpo-k16-cap2500-s0-20260614-r2_step500_full.csv`
- Full-test CI files: `experiments/evidence/H-grpo-k16-cap2500-s0-20260614-r2/eval/H-grpo-k16-cap2500-s0-20260614-r2_full_ci.csv`, `experiments/evidence/H-grpo-k16-cap2500-s0-20260614-r2/eval/H-grpo-k16-cap2500-s0-20260614-r2_full_ci.json`

K=16 n=64 screen: best retained checkpoint step `500` with `31/64` exact; final / last-good checkpoint is also step `500`.

K=16 full-test result:

| Model / checkpoint | Exact | Partial | Format | Empty |
|---|---:|---:|---:|---:|
| Base greedy | `625/1319` (`47.38%`) | `659/1319` (`49.96%`) | `53/1319` (`4.02%`) | `0/1319` |
| K=16 step `500` | `703/1319` (`53.30%`) | `729/1319` (`55.27%`) | `1100/1319` (`83.40%`) | `0/1319` |

Paired bootstrap exact delta vs base, `10000` resamples, seed `12345`: step `500` is `+5.91 pp` with 95% CI `[+3.18, +8.64]`.

## Optional Existing RLOO Full-Test Evidence

The RLOO full-test evals are already available and can be used if useful, but they are not required for Baron's report coherence.

- Consolidated note: `experiments/team_evidence/evidence_harvey_full_eval_20260614.md`
- RLOO K=2 run: `R7-rloo-k2-det-harvey-full-s0-20260611_102009`, branch `harvey-grpo-k8-rerun`, commit `71aab87dee2d2c78256384d084d063d8b40c9e0c`, W&B `https://wandb.ai/barongracias-university-of-cambridge/agentic-ai-coursework/runs/R7-rloo-k2-det-harvey-full-s0-20260611_102009`
- RLOO K=2 evidence: `experiments/evidence/R7-rloo-k2-det-harvey-full-s0-20260611_102009/`
- RLOO K=2 full-test CSVs: `eval/base_full.csv`, `eval/r7_rloo_k2_step500_full.csv`, `eval/r7_rloo_k2_step3364_full.csv`
- RLOO K=8 run: `R7-rloo-k8-det-harvey-full-s0-20260611_105132`, branch `harvey-grpo-k8-rerun`, commit `e3d69a1938fe8e8a2a67a3c84a03331133ed46f1`
- RLOO K=8 evidence: `experiments/evidence/R7-rloo-k8-det-harvey-full-s0-20260611_105132/`
- RLOO K=8 full-test CSVs: `eval/base_full.csv`, `eval/r7_rloo_k8_step2000_full.csv`, `eval/r7_rloo_k8_step3364_full.csv`

RLOO full-test results on the same `07f0f0fc...` manifest: K=2 step `500` is `562/1319`, K=2 final step `3364` is `59/1319` with `1111/1319` empty responses, K=8 step `2000` is `722/1319`, and K=8 final step `3364` is `742/1319`.

## Caveats

- Do not mix the deterministic n=64 screen (`9aa1814e...`) with Baron's older seed-0-draw n=64 evals that had an empty `EVAL_MANIFEST`.
- Do not describe K=16 as completed. It was capped at `2500` planned steps and stopped early with `RESOURCE_EXHAUSTED`; step `500` is the last-good retained checkpoint.
- Runtime is the only requested field not present in the committed lightweight K=4/K=16 local artefacts. The W&B URLs are recorded above; use W&B directly if exact UI runtime is needed.

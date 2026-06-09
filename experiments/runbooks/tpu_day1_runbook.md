# TPU Day-One Runbook

Purpose: get from first TPU access to a verified, reproducible debug run before spending the full 5 hour training budget.

Date target: Monday 2026-06-08.

## Ground Rules

- Do not start a full run until the debug gates below pass.
- Keep checkpoints and TensorBoard logs outside `/tmp`; `/tmp` is not persistent enough for report evidence.
- Record every attempted run in an iteration log before launching it.
- Treat the first TPU session as an integration test unless the debug run is clean.
- Do not push from the TPU VM.

## Inputs

- Coursework PDF requirements and marking criteria.
- `experiments/manifests/i3_sweep_plan.md` for the intended experiment matrix.
- `experiments/manifests/baseline_patch_plan.md` for minimum patches before proper runs.
- `external/tpu-2026/tpu-setup.md` for upstream TPU setup instructions.
- `external/tpu-2026/scripts/README.md` for upstream training/evaluation usage.

## 1. Confirm Local Run Matrix

Before connecting to TPU, decide the first two debug configurations:

| Run | Estimator | Steps | Seed | Goal |
| --- | --- | ---: | ---: | --- |
| D1 | GRPO | 50 | 0 | Verify upstream baseline can train and log. |
| D2 | RLOO | 50 | 0 | Verify estimator switch works with identical controls. |

Full-run candidates should stay aligned with `experiments/manifests/i3_sweep_plan.md`; do not expand the matrix until the first debug evidence is clean.

## 2. TPU Environment Setup

Follow `external/tpu-2026/tpu-setup.md` for project, zone, TPU name, and IAP SSH details. The upstream notes use Python 3.12 through `uv` and create a `~/venvs/tunix` environment.

Minimum secrets/environment values:

```bash
export HF_TOKEN="<huggingface token>"
export WANDB_API_KEY="<wandb token>"
export WANDB_ENTITY="<team or personal entity>"
export WANDB_PROJECT="mas-agentic-ai-tpu-2026"
```

If Kaggle data is used instead of TFDS, also provide the Kaggle credentials expected by the upstream scripts.

## 3. Repository And Environment Bootstrap

On the TPU VM, use the team GitLab repository once available. Until then, the external `tpu-2026` checkout is only the upstream reference.

Inside the upstream training checkout:

```bash
uv python install 3.12
./bootstrap.sh
source ~/venvs/tunix/bin/activate
```

Verify TPU visibility:

```bash
python - <<'PY'
import jax
print("backend:", jax.default_backend())
print("devices:", jax.devices())
PY
```

Expected: backend is TPU and multiple TPU devices are listed.

Verify the Tunix GRPO API supports the required estimator switch:

```bash
python - <<'PY'
from tunix.rl.grpo.grpo_learner import GRPOConfig, GRPOLearner
c = GRPOConfig(num_generations=2, num_iterations=1, beta=0.08, epsilon=0.2)
print(c)
print("advantage_estimator:", getattr(c, "advantage_estimator", None))
print("learner accepts data_shuffle_seed: inspect source if needed")
PY
```

## 4. Apply Minimum Baseline Patches

Apply only the patches required by `experiments/manifests/baseline_patch_plan.md`:

- estimator selection,
- reproducibility seed controls,
- persistent output directories,
- evaluation checkpoint restore,
- per-prompt evaluation output for later confidence intervals.

Do not start a full run until these are verified.

## 5. Create Persistent Run Root

Use a per-run root under `$HOME`, not `/tmp`.

```bash
export RUN_ROOT="$HOME/tpu-runs/$(date +%Y%m%d_%H%M%S)_debug"
mkdir -p "$RUN_ROOT"/{ckpts,intermediate_ckpt,tensorboard,logs,configs,eval}
```

After the baseline patches exist, training should read these paths through environment variables:

```bash
export CKPT_DIR="$RUN_ROOT/ckpts"
export INTERMEDIATE_CKPT_DIR="$RUN_ROOT/intermediate_ckpt"
export TENSORBOARD_DIR="$RUN_ROOT/tensorboard"
export RUN_SEED=0
export MAX_STEPS_OVERRIDE=50
export SAVE_INTERVAL_STEPS=50
```

## 6. Debug Run D1: GRPO

Create an iteration log from `experiments/templates/iteration_log_template.md` before launching.

Suggested launch shape after patching:

```bash
tmux new -s d1-grpo
source ~/venvs/tunix/bin/activate
cd ~/tpu-2026/scripts
export ADV_ESTIMATOR=grpo
python -u train.py --source tfds --wandb-run-id "d1-grpo-s0" 2>&1 | tee "$RUN_ROOT/logs/train_d1_grpo.log"
```

Checks before considering D1 successful:

- JAX backend printed as TPU.
- W&B run appears under the intended entity/project.
- TensorBoard event files appear under `$TENSORBOARD_DIR`.
- Checkpoints appear under `$CKPT_DIR` or `$INTERMEDIATE_CKPT_DIR`; for LoRA restore, check whether Tunix writes actor checkpoints under `$CKPT_DIR/actor`.
- Training logs include reward, KL, and Tunix diagnostic metrics.
- No evidence that output paths silently fell back to `/tmp`.

## 7. Debug Run D2: RLOO

Use the same seed and step budget as D1.

```bash
tmux new -s d2-rloo
source ~/venvs/tunix/bin/activate
cd ~/tpu-2026/scripts
export ADV_ESTIMATOR=rloo
python -u train.py --source tfds --wandb-run-id "d2-rloo-s0" 2>&1 | tee "$RUN_ROOT/logs/train_d2_rloo.log"
```

Additional checks:

- The logged or printed run config states `ADV_ESTIMATOR=rloo`.
- RLOO training reaches the same debug step count as D1.
- Diagnostics can be compared against GRPO without changing unrelated controls.

## 8. Evaluation Restore Check

Before any full run, confirm `evaluate.py` restores a trained LoRA checkpoint. The upstream `chat.py` has restore logic, but upstream `evaluate.py` appears to construct a LoRA model without restoring a checkpoint.

Expected post-patch command shape:

```bash
python -u evaluate.py --preset greedy --source tfds --ckpt-dir "$CKPT_DIR" --output-csv "$RUN_ROOT/eval/d1_grpo_eval.csv"
```

Gate:

- Evaluation output records the requested checkpoint path, resolved checkpoint path, and restored step. If `$CKPT_DIR/actor` exists, the patched evaluator should resolve to it automatically.
- Per-prompt correctness or score data is saved for bootstrap confidence intervals.
- Aggregate numbers are reproducible from the saved per-prompt data.

## 9. Full-Run Gate

Only start full training after all items are true:

- TPU backend verified.
- Python 3.12 environment boots cleanly.
- Estimator switch verified for both GRPO and RLOO.
- Run seed and dataset shuffle seed are controlled.
- Checkpoints and TensorBoard logs are persistent.
- W&B entity/project are correct.
- Evaluation restores trained checkpoints and reports the resolved actor checkpoint root.
- Per-prompt evaluation data is saved.
- Iteration log exists and includes the exact command.

## 10. Stop Conditions

Stop and diagnose instead of launching full training if:

- JAX reports CPU/GPU instead of TPU.
- Outputs go to `/tmp`.
- W&B logs to the wrong account/project.
- `ADV_ESTIMATOR` is ignored or not visible in config output.
- Evaluation cannot identify the checkpoint step being restored.
- Reward/KL/diagnostic metrics are missing.
- The debug run crashes before producing a checkpoint.


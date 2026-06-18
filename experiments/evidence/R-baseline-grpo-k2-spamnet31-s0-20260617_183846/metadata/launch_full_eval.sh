#!/usr/bin/env bash
set -eo pipefail

if [[ -f /home/harvey/.env ]]; then
  set -a
  source /home/harvey/.env
  set +a
fi
set -u

export REPO="/home/harvey/tpu-2026"
export VENV="/home/harvey/venvs/tunix"
export RUN_ID="R-baseline-grpo-k2-spamnet31-s0-20260617_183846"
export RUN_ROOT="/home/harvey/tpu-runs/part-i/R-baseline-grpo-k2-spamnet31-s0-20260617_183846"
export CKPT_DIR="${RUN_ROOT}/ckpts"
export EVAL_DIR="${RUN_ROOT}/eval"
export LOG_DIR="${RUN_ROOT}/logs"
export METADATA_DIR="${RUN_ROOT}/metadata"
export TMPDIR="${RUN_ROOT}/tmp"

export MODEL_REVISION="dcc83ea841ab6100d6b47a070329e1ba4cf78752"
export JAX_REF="3ff6dc40cdb982921de0ef12c0bf8e5c64311a6f"
export TUNIX_REF="7fadb3c81e4348f714b2b5a07f6cc2bd10da10de"
export QWIX_REF="c2548f06eeca005090329530576623afcd48241c"
export FLAX_REF="5ab9d1463550a7fc7fe13575af55081e68258dd7"

export DATA_SOURCE="tfds"
export ADV_ESTIMATOR="grpo"
export NUM_GENERATIONS="2"
export RUN_SEED="0"
export EVAL_SEED="0"
export SCREEN_MANIFEST="/home/harvey/tpu-2026/experiments/evidence/R7-rloo-k2-det-harvey-full-s0-20260611_102009/manifests/gsm8k_test_seed0_n64.jsonl"
export FULL_MANIFEST="/home/harvey/tpu-2026/experiments/manifests/gsm8k_test_seed0_full.jsonl"
export EVAL_MANIFEST="${FULL_MANIFEST}"
export TRAIN_DATA_DIR="${RUN_ROOT}/data/train"
export TEST_DATA_DIR="${RUN_ROOT}/data/test"

export WANDB_PROJECT="agentic-ai-coursework"
export WANDB_ENTITY="barongracias-university-of-cambridge"
export WANDB_RUN_ID="${RUN_ID}"
export WANDB_NAME="${RUN_ID}-eval"
export WANDB_DIR="${RUN_ROOT}/wandb"
export WANDB_MODE="online"
unset MAX_STEPS_OVERRIDE

mkdir -p "${EVAL_DIR}" "${LOG_DIR}" "${METADATA_DIR}" "${TMPDIR}" "${WANDB_DIR}"
source "${VENV}/bin/activate"

SCREEN_HASH_EXPECTED="9aa1814e295e155f4735c9302a7f9c28c6aaa2319074e053a5888e3f3b2448b0"
FULL_HASH_EXPECTED="07f0f0fc10580dee941b6f921a3986854a8b0b74529d9bb952662d5daaea6bb2"
SCREEN_STEPS=(250 500 750 1000 1250 1500 1750 2000 2250 2500 2750 3000 3250 3364)

echo "[$(date -Is)] Starting deterministic baseline eval protocol for ${RUN_ID}"
echo "repo=${REPO}"
echo "run_root=${RUN_ROOT}"
echo "screen_manifest=${SCREEN_MANIFEST}"
echo "full_manifest=${FULL_MANIFEST}"

screen_hash="$(sha256sum "${SCREEN_MANIFEST}" | awk '{print $1}')"
full_hash="$(sha256sum "${FULL_MANIFEST}" | awk '{print $1}')"
screen_rows="$(wc -l < "${SCREEN_MANIFEST}")"
full_rows="$(wc -l < "${FULL_MANIFEST}")"

if [[ "${screen_hash}" != "${SCREEN_HASH_EXPECTED}" ]]; then
  echo "ERROR: screen manifest hash mismatch: ${screen_hash}" >&2
  exit 2
fi
if [[ "${full_hash}" != "${FULL_HASH_EXPECTED}" ]]; then
  echo "ERROR: full manifest hash mismatch: ${full_hash}" >&2
  exit 2
fi
if [[ "${screen_rows}" != "64" ]]; then
  echo "ERROR: screen manifest row count is ${screen_rows}, expected 64" >&2
  exit 2
fi
if [[ "${full_rows}" != "1319" ]]; then
  echo "ERROR: full manifest row count is ${full_rows}, expected 1319" >&2
  exit 2
fi
for step in "${SCREEN_STEPS[@]}"; do
  if [[ ! -d "${CKPT_DIR}/actor/${step}" ]]; then
    echo "ERROR: missing checkpoint ${CKPT_DIR}/actor/${step}" >&2
    exit 2
  fi
done

cat > "${METADATA_DIR}/${RUN_ID}_eval_protocol.env" <<EOF
RUN_ID=${RUN_ID}
RUN_ROOT=${RUN_ROOT}
REPO=${REPO}
MODEL=google/gemma-3-1b-it
MODEL_REVISION=${MODEL_REVISION}
JAX_REF=${JAX_REF}
TUNIX_REF=${TUNIX_REF}
QWIX_REF=${QWIX_REF}
FLAX_REF=${FLAX_REF}
DATA_SOURCE=${DATA_SOURCE}
METHOD=GRPO
ADV_ESTIMATOR=${ADV_ESTIMATOR}
NUM_GENERATIONS=${NUM_GENERATIONS}
RUN_SEED=${RUN_SEED}
EVAL_SEED=${EVAL_SEED}
PRESET=greedy
SCREEN_MANIFEST=${SCREEN_MANIFEST}
SCREEN_MANIFEST_SHA256=${screen_hash}
SCREEN_MANIFEST_ROWS=${screen_rows}
FULL_MANIFEST=${FULL_MANIFEST}
FULL_MANIFEST_SHA256=${full_hash}
FULL_MANIFEST_ROWS=${full_rows}
SCREEN_STEPS=${SCREEN_STEPS[*]}
BEST_TIE_BREAK=lowest_step
BOOTSTRAP_RESAMPLES=10000
BOOTSTRAP_SEED=12345
EOF

cd "${REPO}/scripts"

echo "[$(date -Is)] Evaluating base model on full manifest"
python -u evaluate.py --preset greedy --source tfds --no-restore \
  --eval-manifest "${FULL_MANIFEST}" \
  --output-csv "${EVAL_DIR}/base_full.csv" \
  2>&1 | tee "${LOG_DIR}/eval_base_full.log"

echo "[$(date -Is)] Screening retained checkpoints on n64 manifest"
for STEP in "${SCREEN_STEPS[@]}"; do
  echo "[$(date -Is)] Screening checkpoint step ${STEP}"
  python -u evaluate.py --preset greedy --source tfds \
    --ckpt-dir "${CKPT_DIR}" --step "${STEP}" \
    --eval-manifest "${SCREEN_MANIFEST}" \
    --output-csv "${EVAL_DIR}/${RUN_ID}_step${STEP}_n64.csv" \
    2>&1 | tee "${LOG_DIR}/eval_${RUN_ID}_step${STEP}_n64.log"
done

python - "${RUN_ID}" "${EVAL_DIR}" "${SCREEN_MANIFEST}" "${SCREEN_STEPS[@]}" <<'PY'
import csv
import os
import sys

run_id, eval_dir, expected_manifest, *steps = sys.argv[1:]
records = []
for step_text in steps:
    step = int(step_text)
    path = os.path.join(eval_dir, f"{run_id}_step{step}_n64.csv")
    with open(path, newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    if len(rows) != 64:
        raise SystemExit(f"{path}: expected 64 rows, found {len(rows)}")
    manifests = {row.get("eval_manifest") for row in rows}
    if manifests != {expected_manifest}:
        raise SystemExit(f"{path}: eval_manifest mismatch: {sorted(manifests)}")
    restored = {row.get("restored_step") for row in rows}
    if restored != {str(step)}:
        raise SystemExit(f"{path}: restored_step mismatch: {sorted(restored)}")
    correct = sum(int(float(row["correct"])) for row in rows)
    partial = sum(int(float(row["partial_correct"])) for row in rows)
    fmt = sum(int(float(row["format_correct"])) for row in rows)
    records.append({
        "step": step,
        "n": len(rows),
        "correct": correct,
        "acc": correct / len(rows),
        "partial_correct": partial,
        "format_correct": fmt,
        "path": path,
    })

records.sort(key=lambda row: (-row["correct"], row["step"]))
best = records[0]
ordered = sorted(records, key=lambda row: row["step"])

metrics_path = os.path.join(eval_dir, f"{run_id}_n64_screen_metrics.csv")
with open(metrics_path, "w", newline="", encoding="utf-8") as fh:
    writer = csv.DictWriter(
        fh,
        fieldnames=["step", "n", "correct", "acc", "partial_correct", "format_correct", "path"],
    )
    writer.writeheader()
    writer.writerows(ordered)

summary_path = os.path.join(eval_dir, f"{run_id}_n64_screen_summary.txt")
with open(summary_path, "w", encoding="utf-8") as fh:
    fh.write("n64 checkpoint screen, greedy, exact-correct selection\n")
    fh.write("tie_break=lowest_step\n")
    for row in ordered:
        fh.write(
            f"step={row['step']} correct={row['correct']}/{row['n']} "
            f"acc={row['acc']:.6f} partial={row['partial_correct']} "
            f"format={row['format_correct']}\n"
        )
    fh.write(f"BEST_STEP={best['step']}\n")
    fh.write(f"BEST_CORRECT={best['correct']}\n")

env_path = os.path.join(eval_dir, f"{run_id}_selected_steps.env")
with open(env_path, "w", encoding="utf-8") as fh:
    fh.write(f"BEST_STEP={best['step']}\n")
    fh.write("FINAL_STEP=3364\n")
    fh.write("BEST_SELECTION_METRIC=exact_correct_n64\n")
    fh.write("BEST_TIE_BREAK=lowest_step\n")

print(f"Selected BEST_STEP={best['step']} by n64 exact correct ({best['correct']}/64)")
print(f"Wrote {metrics_path}")
print(f"Wrote {summary_path}")
print(f"Wrote {env_path}")
PY

source "${EVAL_DIR}/${RUN_ID}_selected_steps.env"

echo "[$(date -Is)] Full-evaluating selected best checkpoint step ${BEST_STEP}"
python -u evaluate.py --preset greedy --source tfds \
  --ckpt-dir "${CKPT_DIR}" --step "${BEST_STEP}" \
  --eval-manifest "${FULL_MANIFEST}" \
  --output-csv "${EVAL_DIR}/${RUN_ID}_step${BEST_STEP}_full.csv" \
  2>&1 | tee "${LOG_DIR}/eval_${RUN_ID}_step${BEST_STEP}_full.log"

if [[ "${BEST_STEP}" == "3364" ]]; then
  echo "[$(date -Is)] Best checkpoint is final checkpoint; reusing step3364 full CSV for both labels"
else
  echo "[$(date -Is)] Full-evaluating final checkpoint step 3364"
  python -u evaluate.py --preset greedy --source tfds \
    --ckpt-dir "${CKPT_DIR}" --step 3364 \
    --eval-manifest "${FULL_MANIFEST}" \
    --output-csv "${EVAL_DIR}/${RUN_ID}_step3364_full.csv" \
    2>&1 | tee "${LOG_DIR}/eval_${RUN_ID}_step3364_full.log"
fi

python - "${RUN_ID}" "${EVAL_DIR}" "${FULL_MANIFEST}" "${BEST_STEP}" <<'PY'
import csv
import json
import os
import sys

run_id, eval_dir, full_manifest, best_step = sys.argv[1:]
checks = [
    ("base", os.path.join(eval_dir, "base_full.csv"), "", 1319),
    ("baseline_best", os.path.join(eval_dir, f"{run_id}_step{best_step}_full.csv"), best_step, 1319),
    ("baseline_final", os.path.join(eval_dir, f"{run_id}_step3364_full.csv"), "3364", 1319),
]
out = []
for label, path, expected_step, expected_rows in checks:
    with open(path, newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    if len(rows) != expected_rows:
        raise SystemExit(f"{path}: expected {expected_rows} rows, found {len(rows)}")
    manifests = {row.get("eval_manifest") for row in rows}
    if manifests != {full_manifest}:
        raise SystemExit(f"{path}: eval_manifest mismatch: {sorted(manifests)}")
    restored = {row.get("restored_step") for row in rows}
    if expected_step == "":
        if restored - {"", "None"}:
            raise SystemExit(f"{path}: expected base/no restored step, got {sorted(restored)}")
    elif restored != {expected_step}:
        raise SystemExit(f"{path}: restored_step mismatch: expected {expected_step}, got {sorted(restored)}")
    correct = sum(int(float(row["correct"])) for row in rows)
    out.append({
        "label": label,
        "path": path,
        "rows": len(rows),
        "correct": correct,
        "accuracy": correct / len(rows),
        "restored_step_values": sorted(restored),
        "eval_manifest": full_manifest,
    })

validation_path = os.path.join(eval_dir, f"{run_id}_eval_validation.json")
with open(validation_path, "w", encoding="utf-8") as fh:
    json.dump(out, fh, indent=2)
print(f"Wrote {validation_path}")
for row in out:
    print(f"{row['label']}: correct={row['correct']}/{row['rows']} acc={row['accuracy']:.6f}")
PY

cd "${REPO}"
echo "[$(date -Is)] Computing paired bootstrap confidence intervals"
python -u experiments/analysis/bootstrap_ci.py \
  --csv base="${EVAL_DIR}/base_full.csv" \
  --csv baseline_best="${EVAL_DIR}/${RUN_ID}_step${BEST_STEP}_full.csv" \
  --csv baseline_final="${EVAL_DIR}/${RUN_ID}_step3364_full.csv" \
  --baseline base \
  --resamples 10000 --seed 12345 \
  --output-json "${EVAL_DIR}/${RUN_ID}_full_ci.json" \
  --output-csv "${EVAL_DIR}/${RUN_ID}_full_ci.csv" \
  2>&1 | tee "${LOG_DIR}/bootstrap_${RUN_ID}_full_ci.log"

echo "[$(date -Is)] Eval protocol complete for ${RUN_ID}"

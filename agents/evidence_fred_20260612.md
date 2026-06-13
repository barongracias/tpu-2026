# Evidence: Fred Experiments, 2026-06-12

## Summary

Local evidence shows completed hard-medium GRPO runs R7 (K=2) and R8 (K=8), a completed hard-question mining pass, and a completed R9 two-stage RLOO K=8 run. R8 has a report-usable final-checkpoint greedy eval on the shared 64-prompt manifest. R7 is usable only as a documented negative/collapse run unless a trained-checkpoint eval is added. R9 now has `scripts/evaluate.py` greedy evals for base, stage 1, and retained stage-2 checkpoints through step 3364.

W&B was not queried directly in this audit. URLs below are taken from local logs and notes.

## Git State

- repo path: `/home/fredlawrence/tpu-2026`
- current branch: `fred`
- current HEAD: `a93ea9f387b8c5083607263daf157929482c50cb` (`hard data + runs + evals`)
- status: dirty. Modified: `experiments/README.md`, `scripts/config.py`, `scripts/rewards.py`, `scripts/train.py`. Untracked evidence/run-note files include this audit prompt, hard-medium evidence, R7/R8/R9 notes, and R9 launcher scripts.
- fetch: `git fetch --all --prune` was run after approval because `.git/FETCH_HEAD` is read-only in the sandbox. It updated `origin/harvey-grpo-k8-rerun` to `f3da2c8`.
- relevant branches: `fred`, `main`, `baron`, `origin/fred`, `origin/deterministic-platform`, `origin/coursework`, `origin/baron`, `origin/baron_k8`, `origin/harvey`, `origin/harvey-grpo-k8-rerun`, `origin/main`.
- relevant commits: `a93ea9f` current Fred evidence/code notes; `57c6409` deterministic-platform launch base for R7/R8/mining; `f3da2c8` latest Harvey rerun artifacts; `7ab0f1e` Baron branch; `8a0f7f2` Harvey branch.

## Experiment Register

| Run | Branch | Commit | Estimator | K | Seed | Steps | Status | Best exact | Final exact | W&B | Artefacts usable? |
| --- | --- | --- | --- | ---: | ---: | ---: | --- | ---: | ---: | --- | --- |
| hard_mining_base_gsm8k_train_20260609_220117 | fred | 57c6409 + dirty/untracked mining script later in a93ea9f | base probe | n/a | 0 | n/a | complete | n/a | n/a | none found locally | yes, for manifest provenance |
| R7-grpo-k2-hardmedium-20260610_102859 | fred | 57c6409 + dirty/untracked manifest-training changes later in a93ea9f | GRPO | 2 | 0 | 3452 | complete; collapsed late | n/a | not evaluated as trained ckpt | 3v8bx0iz | partial; not report-ready for accuracy |
| R8-grpo-k8-hardmedium-20260610_131135 | fred | 57c6409 + dirty/untracked manifest-training changes later in a93ea9f | GRPO | 8 | 0 | 3452 | complete | n/a; no retained sweep | 46.88 | xnn69ylg | yes, final-checkpoint only |
| R9-rloo-k8-format-answer-full-20260612_131553 | fred | a93ea9f | RLOO | 8 | 0 | 500 stage 1; 3364 stage 2 | complete; evals done with `scripts/evaluate.py` | 57.81 at steps 2500/3000 | 53.12 at step 3364 | 2f7wikq5, izp0f1u1 | yes, with checkpoint-selection caveat |

## Run Details

### hard_mining_base_gsm8k_train_20260609_220117

Identity/config/runtime:
- Run root: `/home/fredlawrence/tpu-runs/part-i/hard_mining_base_gsm8k_train_20260609_220117`
- Repo evidence: `experiments/evidence/hard_mining_base_gsm8k_train_20260609_220117/`
- Created: `2026-06-09T22:01:17.746597+00:00`
- Metadata launch commit: `fred`, `57c6409add0d81bbdb32ca7f4b3e176b4e044068`
- Dirty at launch: yes. Metadata records modified `agents/context.md`, `agents/report_notes.md`, `experiments/manifests/experiment_contract.md`, and untracked `scripts/mine_hard_examples.py`.
- Command: `/home/fredlawrence/venvs/tunix/bin/python scripts/mine_hard_examples.py`
- Platform: personal `federico-tpu` VM per user statement; exact TPU type/zone not found locally.

Artefacts/metrics:
- Metadata: `experiments/evidence/hard_mining_base_gsm8k_train_20260609_220117/run_metadata.json`
- Summary: `experiments/evidence/hard_mining_base_gsm8k_train_20260609_220117/summary.md`
- Local outputs: `all_examples.csv`, `all_examples.jsonl`, `easy_examples.jsonl`, `medium_examples.jsonl`, `hard_examples.jsonl`, and per-example completions under the run root.
- Processed 7,473 GSM8K train examples: 3,637 easy, 1,990 medium, 1,846 hard, 0 skipped.
- Training manifest: `experiments/evidence/manifests/gsm8k_train_base_hard_medium.jsonl`, 3,836 rows.
- Eval manifest: `experiments/evidence/manifests/gsm8k_test_seed0_n64.jsonl`, 64 rows.

Caveats:
- W&B is not applicable/not found locally for this inference-only mining pass.
- The mining script itself was untracked at launch, but its current copy is present in the repo.

### R7-grpo-k2-hardmedium-20260610_102859

Identity/config/runtime:
- Nickname/run id: `R7-grpo-k2-hardmedium-20260610_102859`
- W&B: `https://wandb.ai/barongracias-university-of-cambridge/agentic-ai-coursework/runs/3v8bx0iz`
- Run root: `/home/fredlawrence/tpu-runs/part-i/R7-grpo-k2-hardmedium-20260610_102859`
- Metadata launch commit: `fred`, `57c6409add0d81bbdb32ca7f4b3e176b4e044068`
- Provenance caveat: do not report this as a clean `57c6409` run. The
  manifest-training support used by this run was absent from clean `57c6409`;
  describe it as `57c6409` plus dirty/untracked changes later committed as
  `a93ea9f387b8c5083607263daf157929482c50cb`. The exact launch diff is not
  proven locally.
- Launch timestamp: `2026-06-10T10:29:04.818488Z` in `ckpts/run_metadata.json`
- Estimator/K/seed: `ADV_ESTIMATOR=grpo`, `NUM_GENERATIONS=2`, `RUN_SEED=0`, `EVAL_SEED=0`
- Steps/checkpointing: `max_steps=3452`, `SAVE_INTERVAL_STEPS=250`, `MAX_TO_KEEP=20`
- Data: `DATA_SOURCE=manifest`; train manifest `/home/fredlawrence/tpu-runs/part-i/manifests/gsm8k_train_base_hard_medium.jsonl`
- Eval manifest: metadata has empty `eval_manifest`; later notes point to shared `/home/fredlawrence/tpu-runs/part-i/manifests/gsm8k_test_seed0_n64.jsonl`
- Platform: personal `federico-tpu` VM per user statement; exact TPU type/zone not found locally.

Artefacts:
- Metadata: `experiments/evidence/R7-grpo-k2-hardmedium-20260610_102859/run_metadata.json`
- Training log: `/home/fredlawrence/tpu-runs/part-i/R7-grpo-k2-hardmedium-20260610_102859/logs/train.log`
- TensorBoard event: `/home/fredlawrence/tpu-runs/part-i/R7-grpo-k2-hardmedium-20260610_102859/tensorboard/events.out.tfevents.1781087357.t1v-n-b6cf1858-w-0`
- TensorBoard scalar export: `experiments/evidence/R7-grpo-k2-hardmedium-20260610_102859/tensorboard_scalars.csv`
- TensorBoard scalar summary: `experiments/evidence/R7-grpo-k2-hardmedium-20260610_102859/tensorboard_scalar_summary.json`
- Checkpoints retained under `ckpts/actor`: `1`, `250`, `500`, `750`, `1000`, `1250`, `1500`, `1750`, `2000`, `2250`, `2500`, `2750`, `3000`, `3250`, `3452`.
- Eval CSV copied to repo: `experiments/evidence/R7-grpo-k2-hardmedium-20260610_102859/r7_best_greedy.csv`

Metrics:
- `r7_best_greedy.csv`: exact `31/64` (48.44%), partial `31/64` (48.44%), format `1/64` (1.56%), empty `0/64`, restored step empty.
- Restore was not confirmed for this CSV: `requested_ckpt_dir=/tmp/content/ckpts/`, `resolved_ckpt_dir=/tmp/content/ckpts/`, `restored_step` empty.
- No trained R7 checkpoint eval was found locally.

Caveats:
- Treat `r7_best_greedy.csv` as a base-reference/mixup artefact, not a trained R7 result.
- Training log ends with `Training finished`, but late log samples include empty responses; local notes describe collapse from around step 518.
- W&B warnings show repeated out-of-order step logging (`step 0` less than current step `3452`), so TensorBoard should be preferred for scalar curves.
- Report status: usable as a negative/collapse run with local scalar evidence.
  A trained salvage checkpoint eval such as step 250 is still needed if this
  run should contribute an accuracy row.

### R8-grpo-k8-hardmedium-20260610_131135

Identity/config/runtime:
- Nickname/run id: `R8-grpo-k8-hardmedium-20260610_131135`
- W&B: `https://wandb.ai/barongracias-university-of-cambridge/agentic-ai-coursework/runs/xnn69ylg`
- Run root: `/home/fredlawrence/tpu-runs/part-i/R8-grpo-k8-hardmedium-20260610_131135`
- Metadata launch commit: `fred`, `57c6409add0d81bbdb32ca7f4b3e176b4e044068`
- Provenance caveat: do not report this as a clean `57c6409` run. The
  manifest-training support used by this run was absent from clean `57c6409`;
  describe it as `57c6409` plus dirty/untracked changes later committed as
  `a93ea9f387b8c5083607263daf157929482c50cb`. The exact launch diff is not
  proven locally.
- Launch timestamp: `2026-06-10T13:11:40.249392Z` in `ckpts/run_metadata.json`
- Estimator/K/seed: `ADV_ESTIMATOR=grpo`, `NUM_GENERATIONS=8`, `RUN_SEED=0`, `EVAL_SEED=0`
- Steps/checkpointing: `max_steps=3452`, `SAVE_INTERVAL_STEPS=250`, `MAX_TO_KEEP=20`
- Data: `DATA_SOURCE=manifest`; train manifest `/home/fredlawrence/tpu-runs/part-i/manifests/gsm8k_train_base_hard_medium.jsonl`
- Eval manifest: final eval log confirms `/home/fredlawrence/tpu-runs/part-i/manifests/gsm8k_test_seed0_n64.jsonl`
- Platform: personal `federico-tpu` VM per user statement; exact TPU type/zone not found locally.

Artefacts:
- Metadata: `experiments/evidence/R8-grpo-k8-hardmedium-20260610_131135/run_metadata.json`
- Training log: `/home/fredlawrence/tpu-runs/part-i/R8-grpo-k8-hardmedium-20260610_131135/logs/train.log`
- Eval logs: `logs/eval_r8_step3452_greedy.log`, `logs/eval_best_greedy.log`
- TensorBoard event: `/home/fredlawrence/tpu-runs/part-i/R8-grpo-k8-hardmedium-20260610_131135/tensorboard/events.out.tfevents.1781097112.t1v-n-b6cf1858-w-0`
- TensorBoard scalar export: `experiments/evidence/R8-grpo-k8-hardmedium-20260610_131135/tensorboard_scalars.csv`
- TensorBoard scalar summary: `experiments/evidence/R8-grpo-k8-hardmedium-20260610_131135/tensorboard_scalar_summary.json`
- Checkpoints retained under `ckpts/actor`: `1`, `250`, `500`, `750`, `1000`, `1250`, `1500`, `1750`, `2000`, `2250`, `2500`, `2750`, `3000`, `3250`, `3452`.
- Repo eval CSVs: `experiments/evidence/R8-grpo-k8-hardmedium-20260610_131135/r8_step3452_greedy.csv`, `experiments/evidence/R8-grpo-k8-hardmedium-20260610_131135/best_greedy.csv`

Metrics:
- Final trained eval, `r8_step3452_greedy.csv`: restored step `3452`; exact `30/64` (46.88%), partial `32/64` (50.00%), format `60/64` (93.75%), empty `0/64`; restore confirmed from `ckpts/actor`.
- Base/mixup eval, `best_greedy.csv`: exact `31/64` (48.44%), partial `31/64` (48.44%), format `1/64` (1.56%), empty `0/64`; `restored_step` empty, so not a trained checkpoint result.
- Base comparison on same 64-prompt eval sample exists via the mixup/base CSV.

Caveats:
- Training log ends with `Training finished`.
- W&B warnings show repeated out-of-order step logging (`step 0` less than current step `3452`), so TensorBoard should be preferred for scalar curves.
- Earlier retained checkpoint evals such as 500/1500/2500 were not found locally; this limits best-checkpoint selection.
- Report status: currently usable for the final report as a final-checkpoint
  hard-medium GRPO K=8 result, with the caveat that final underperforms the
  local base reference on exact accuracy. Do not call the `30/64` result a
  best retained checkpoint; no retained-checkpoint sweep is locally evidenced.

### R9-rloo-k8-format-answer-full-20260612_131553

Identity/config/runtime:
- Nickname/run id: `R9-rloo-k8-format-answer-full-20260612_131553`
- W&B stage 1: `https://wandb.ai/barongracias-university-of-cambridge/agentic-ai-coursework/runs/2f7wikq5`
- W&B stage 2: `https://wandb.ai/barongracias-university-of-cambridge/agentic-ai-coursework/runs/izp0f1u1`
- Run root: `/home/fredlawrence/tpu-runs/part-i/R9-rloo-k8-format-answer-full-20260612_131553`
- Branch/commit: `fred`, `a93ea9f387b8c5083607263daf157929482c50cb`
- Dirty launch status: launch metadata does not include `git status`; current worktree has uncommitted reward-profile and launcher changes used by this run.
- Launch commit file: `/home/fredlawrence/tpu-runs/part-i/R9-rloo-k8-format-answer-full-20260612_131553/logs/launch_commit.txt`
- Estimator/K/seed: `ADV_ESTIMATOR=rloo`, `NUM_GENERATIONS=8`, `RUN_SEED=0`, `EVAL_SEED=0`
- Stage 1: `REWARD_PROFILE=format_heavy`, `MAX_STEPS_OVERRIDE=500`
- Stage 2: `REWARD_PROFILE=answer_heavy`, `MAX_STEPS_OVERRIDE=3364`
- Data: `DATA_SOURCE=tfds`, `NUM_BATCHES=3738`, run-local train/test data dirs.
- Eval manifest: `/home/fredlawrence/tpu-runs/part-i/manifests/gsm8k_test_seed0_n64.jsonl`
- Platform: personal `federico-tpu` VM per user statement; exact TPU type/zone not found locally.

Artefacts:
- Launch env: `logs/launch_env.txt`
- Stage 1 metadata: `logs/run_metadata_stage1_format.json`
- Current checkpoint metadata: `ckpts/run_metadata.json`, overwritten by the stage 2 run and records `reward_profile=answer_heavy`
- Training logs: `logs/train_stage1_format.log`, `logs/train_stage2_answer.log`
- Eval script: all listed eval CSVs/logs were produced by `scripts/evaluate.py` via `scripts/run_rloo_k8_format_then_answer_full.sh` `run_eval()`.
- Repo evidence directory: `experiments/evidence/R9-rloo-k8-format-answer-full-20260612_131553/`
- Repo eval CSVs: `base_greedy.csv`, `stage1_step500_greedy.csv`, `stage2_step500_greedy.csv`, `stage2_step1000_greedy.csv`, `stage2_step2000_greedy.csv`, `stage2_step2500_greedy.csv`, `stage2_step3000_greedy.csv`, `stage2_step3364_greedy.csv`
- Repo metadata/summaries: `run_metadata_stage1_format.json`, `run_metadata_stage2_answer.json`, `base_summary.txt`, `stage1_summary.txt`, `final_eval_summary.txt`, `launch_commit.txt`
- Eval logs: `logs/eval_base_greedy.log`, `logs/eval_stage1_step500_greedy.log`, `logs/eval_stage2_step500_greedy.log`, `logs/eval_stage2_step1000_greedy.log`, `logs/eval_stage2_step2000_greedy.log`, `logs/eval_stage2_step2500_greedy.log`, `logs/eval_stage2_step3000_greedy.log`, `logs/eval_stage2_step3364_greedy.log`
- TensorBoard event: `/home/fredlawrence/tpu-runs/part-i/R9-rloo-k8-format-answer-full-20260612_131553/tensorboard/events.out.tfevents.1781270249.t1v-n-b6cf1858-w-0`
- TensorBoard scalar export: `experiments/evidence/R9-rloo-k8-format-answer-full-20260612_131553/tensorboard_scalars.csv`
- TensorBoard scalar summary: `experiments/evidence/R9-rloo-k8-format-answer-full-20260612_131553/tensorboard_scalar_summary.json`
- Checkpoints retained under `ckpts/actor`: `1`, `250`, `500`, `750`, `1000`, `1250`, `1500`, `1750`, `2000`, `2250`, `2500`, `2750`, `3000`, `3250`, `3364`.

Metrics:
- Base eval: exact `31/64` (48.44%), partial `31/64` (48.44%), format `1/64` (1.56%), empty `0/64`, no restore.
- Stage 1 step 500 eval: restored step `500`; exact `30/64` (46.88%), partial `36/64` (56.25%), format `60/64` (93.75%), empty `0/64`; restore confirmed from `ckpts/actor`.
- Stage 1 gate passed locally (`format=93.75%`, threshold `90%`; empty `0`, max `4`).
- Stage 2 retained-checkpoint greedy evals from `scripts/evaluate.py`:

| Step | Exact | Partial | Format | Empty |
| ---: | ---: | ---: | ---: | ---: |
| 500 | 30/64 (46.88%) | 36/64 (56.25%) | 60/64 (93.75%) | 0/64 |
| 1000 | 30/64 (46.88%) | 31/64 (48.44%) | 62/64 (96.88%) | 0/64 |
| 2000 | 33/64 (51.56%) | 35/64 (54.69%) | 56/64 (87.50%) | 0/64 |
| 2500 | 37/64 (57.81%) | 37/64 (57.81%) | 58/64 (90.62%) | 0/64 |
| 3000 | 37/64 (57.81%) | 38/64 (59.38%) | 57/64 (89.06%) | 0/64 |
| 3364 | 34/64 (53.12%) | 36/64 (56.25%) | 56/64 (87.50%) | 0/64 |

- Best retained R9 checkpoint by exact accuracy is a tie at steps `2500` and `3000` (`37/64`, 57.81%). Step `3000` has the best partial score (`38/64`, 59.38%). Final step `3364` is lower (`34/64`, 53.12%) but still above the local base reference (`31/64`).

Caveats:
- An earlier visible stage-2 attempt hit `AttributeError: 'google._upb._message.FieldDescriptor' object has no attribute 'label'` while constructing TFDS `gsm8k`; the later/retry stage-2 run in the same `131553` root completed training and wrote final evals.
- A prior R9 root, `/home/fredlawrence/tpu-runs/part-i/R9-rloo-k8-format-answer-full-20260612_131040`, exists with base evidence and stage-1 log output, but the richer/continued root is `131553`.
- Report status: R9 can now support a completed two-stage RLOO K=8 accuracy claim, with a checkpoint-selection caveat because the best retained checkpoints are steps `2500`/`3000`, not final step `3364`.

## Hard-Question Mining

- Hard-question mining was used for R7 and R8.
- Method: frozen base `google/gemma-3-1b-it@dcc83ea841ab6100d6b47a070329e1ba4cf78752` on GSM8K train, one greedy completion plus 8 sampled standard completions at seeds `0..7`.
- Source baseline: base model, no LoRA checkpoint restore.
- Manifest path: `experiments/evidence/manifests/gsm8k_train_base_hard_medium.jsonl`; local source path `/home/fredlawrence/tpu-runs/part-i/manifests/gsm8k_train_base_hard_medium.jsonl`.
- Number of questions: 3,836 hard+medium train examples; shared eval manifest has 64 test prompts.
- Criteria for hard: greedy incorrect and zero correct sampled completions. Criteria for medium: greedy incorrect but at least one sampled completion correct. Easy examples were excluded.
- Branch/commit containing code: current pushed Fred branch `a93ea9f` includes `scripts/mine_hard_examples.py`; mining metadata records launch commit `57c6409` with the script untracked.
- Runs using manifest: R7 GRPO K=2 hard-medium and R8 GRPO K=8 hard-medium. The GRPO R9 hard-medium plan exists, but no completed GRPO R9 run root was found locally.

## Missing Evidence Checklist

- R7 trained-checkpoint eval CSV, especially a salvage eval at `ckpts/actor/250`.
- R8 earlier retained checkpoint evals, for example 500, 1500, and 2500, if best-checkpoint selection is needed.
- Exact TPU type and zone for the personal `federico-tpu` VM.
- Direct W&B runtime/status confirmation, if W&B access is needed; this audit used only local logs and notes.
- Dirty worktree status at R7/R8 launch; current metadata does not record it.

## Report-Ready Claims

- Fred produced a hard+medium GSM8K train manifest with 3,836 examples selected from 7,473 train examples using a frozen base-model probe.
- The shared held-out eval manifest contains 64 GSM8K test prompts.
- R8 GRPO K=8 hard-medium completed to step 3452 and restored successfully for final greedy eval on the shared 64-prompt manifest.
- R8 final trained checkpoint achieved exact `30/64` (46.88%), partial `32/64` (50.00%), format `60/64` (93.75%), empty `0/64`.
- The available base/reference eval on the same 64 prompts is exact `31/64` (48.44%), partial `31/64` (48.44%), format `1/64` (1.56%), empty `0/64`.
- R9 RLOO K=8 two-stage run stage 1 restored at step 500 and achieved exact `30/64` (46.88%), partial `36/64` (56.25%), format `60/64` (93.75%), empty `0/64`.
- R9 stage 2 completed and was evaluated with `scripts/evaluate.py` on the shared 64-prompt manifest. Best retained checkpoints were steps `2500` and `3000` at exact `37/64` (57.81%); final step `3364` achieved exact `34/64` (53.12%), partial `36/64` (56.25%), format `56/64` (87.50%), empty `0/64`.

## Not Yet Report-Ready

- R7 trained accuracy or best checkpoint, because the only copied CSV is a base/mixup eval with no restored step.
- R7 collapse severity as a plotted diagnostic, until TensorBoard scalars are exported or consumed.
- Any claim that hard-medium GRPO K=2 was better/worse than K=8 on trained accuracy; R7 trained eval is missing.
- Any strong claim that R9 two-stage reward scheduling improves general performance beyond this 64-prompt sample; paired/bootstrap uncertainty is still needed.
- Any W&B runtime or final remote status claim; W&B was not queried directly in this audit.

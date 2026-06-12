# Evidence: Codex Experiments, 2026-06-12

## Summary

Local evidence supports a Baron experiment sequence of D1/D2 debug runs, R1/R3 K=2 full runs, D3/D4 K=8 diagnostics, and R5 GRPO K=8 full seed 0. R5 is currently the strongest report-usable trained run: best retained checkpoint step 3250 reached 35/64 exact on the local 64-prompt greedy eval, while final step 3364 reached 32/64. R1 and R3 are usable as negative/comparison evidence with caveats. D1/D2/D3/D4 are mainly operational or diagnostic, not final performance claims.

W&B was not queried. W&B URLs below are taken from local logs, metadata, and notes only.

The requested `baron_k8` branch exists, but locally it points at the same commit as the current deterministic/baron checkout (`57c6409`). I found no separate local evidence that `baron_k8` itself launched a distinct personal-TPU run. K=8 Baron evidence exists locally as D3/D4/R5 run roots. Separately, `origin/baron` contains a 2026-06-12 note asking Harvey for K=8/RLOO artefacts; those Harvey runs are not locally verified here.

## Git State

- repo path: `/home/ext_barongracias_gmail_com/tpu-2026`
- remote: `git@github.com:barongracias/tpu-2026.git`
- current branch: `baron`
- current HEAD: `57c6409add0d81bbdb32ca7f4b3e176b4e044068`
- status before this evidence file was created: `## baron...origin/baron [behind 1]`
- relevant branches:
  - local `baron`: `57c6409add0d81bbdb32ca7f4b3e176b4e044068`
  - `origin/baron`: `7ab0f1e` (`team tasj`), one commit ahead with `agents/team_actions_fred_harvey_20260612.md`
  - `origin/baron_k8`: `57c6409add0d81bbdb32ca7f4b3e176b4e044068`
  - `origin/coursework` and `origin/deterministic-platform`: `57c6409add0d81bbdb32ca7f4b3e176b4e044068`
- relevant commits:
  - `4339ba84ba3396d9e6defaef218ecded435fc787`: R1 metadata launch commit.
  - `99cb7f88b0e49a7d65bb0c5274734a190833aa3f`: R3/D3/D4 metadata launch commit.
  - `820fad61060a4260184e71066568af7b28d3109e`: R5 launch commit.
  - `57c6409add0d81bbdb32ca7f4b3e176b4e044068`: current local branch head, deterministic manifest code.

## Experiment Register

| Run | Branch | Commit | Estimator | K | Seed | Steps | Status | Best exact | Final exact | W&B | Artefacts usable? |
| --- | --- | --- | --- | ---: | ---: | ---: | --- | ---: | ---: | --- | --- |
| D1-grpo-debug-seed0 | coursework/baron lineage | e3ebeef4a3be9ad978959b66d4ecb16deecccefb | GRPO | 2 | 0 | 50 | completed debug | 46.88 | 46.88 | URL in logs | Debug only |
| D2-rloo-debug-seed0 | coursework/baron lineage | be631b2 inferred; metadata says unknown for successful retry | RLOO | 2 | 0 | 50 | completed debug after failed attempt | 46.88 | 46.88 | URL in logs | Debug only |
| R1-grpo-full-s0 | coursework/baron lineage | 4339ba84ba3396d9e6defaef218ecded435fc787 | GRPO | 2 | 0 | 3364 | completed, negative result | 37.50 | 18.75 | URL in logs | Yes, with caveats |
| R3-rloo-full-s0-20260608_230810 | coursework/baron lineage | 99cb7f88b0e49a7d65bb0c5274734a190833aa3f | RLOO | 2 | 0 | 3364 | completed, collapse | 9.38 | 1.56 | URL in logs | Yes, negative/comparison |
| D3-grpo-k8-debug-20260609_094815 | TPU-side patch on 99cb7f8 | 99cb7f88b0e49a7d65bb0c5274734a190833aa3f | GRPO | 8 | 0 | 50 | completed debug | 48.44 | 48.44 | URL in logs | Diagnostic only |
| D4-grpo-k8-medium-debug-20260609_100945 | TPU-side patch on 99cb7f8 | 99cb7f88b0e49a7d65bb0c5274734a190833aa3f | GRPO | 8 | 0 | 500 | completed medium diagnostic | 53.12 | 53.12 | URL in logs | Diagnostic, useful context |
| R5-grpo-k8-full-s0-20260609_114832 | baron/coursework lineage | 820fad61060a4260184e71066568af7b28d3109e | GRPO | 8 | 0 | 3364 | completed | 54.69 | 50.00 | URL in logs | Yes, main positive result |
| R6-grpo-k2-baseline-s0-20260609_212708 | baron deterministic lineage | 57c6409add0d81bbdb32ca7f4b3e176b4e044068 | GRPO | 2 | 0 | 3364 | training completed, no local eval found | n/a | n/a | URL in logs | Not yet |

## Run Details

### D1-grpo-debug-seed0

- Identity: Baron TPU-side runner; run root `/home/ext_barongracias_gmail_com/tpu-runs/part-i/D1-grpo-debug-20260608_142021`; launched `2026-06-08T14:20:38.487254Z`; metadata path `ckpts/run_metadata.json`; branch noted as `coursework`.
- Configuration: GRPO, K inferred default 2, seed 0, max steps 50, save interval 50, dataset `tfds`, greedy eval preset.
- Runtime/platform: Google TPU VM per notes; train log contains `Training finished.`; wall-clock duration not extracted locally.
- Artefacts: train log `logs/train.log`; TensorBoard event `tensorboard/events.out.tfevents.1780928465.t1v-n-0339f27d-w-0`; checkpoint `ckpts/actor/50`; eval CSV `eval/eval_greedy.csv`; eval log `logs/eval_greedy.log`.
- Metrics: step 50, exact 30/64 (46.88%), partial 32/64 (50.00%), format 4/64 (6.25%), empty 0/64, sample size 64, restore confirmed at step 50. No same-run base CSV found locally.
- Caveats: debug only; eval manifest field empty; TFDS/protobuf cache issue noted before successful eval; W&B step-order warnings noted.

### D2-rloo-debug-seed0

- Identity: Baron TPU-side runner; successful run root `/home/ext_barongracias_gmail_com/tpu-runs/part-i/D2-rloo-debug-20260608_144102`; initial failed root `/home/ext_barongracias_gmail_com/tpu-runs/part-i/D2-rloo-debug-20260608_143942`; launched `2026-06-08T14:41:21.543974Z`.
- Configuration: RLOO, K inferred default 2, seed 0, max steps 50, save interval 50, dataset `tfds`, greedy eval preset.
- Runtime/platform: Google TPU VM per notes; successful train log completed. Initial attempt failed due TFDS/protobuf metadata cache issue.
- Artefacts: train log `logs/train.log`; TensorBoard event path in notes; checkpoint `ckpts/actor/50`; eval CSV `eval/eval_greedy.csv`; eval log `logs/eval_greedy.log`.
- Metrics: step 50, exact 30/64 (46.88%), partial 30/64 (46.88%), format 3/64 (4.69%), empty 0/64, sample size 64, restore confirmed at step 50. No same-run base CSV found locally.
- Caveats: metadata commit is `unknown` for successful retry because it ran from run-local cwd; use synced HEAD `be631b2` from notes as inferred provenance. Debug only.

### R1-grpo-full-s0

- Identity: Baron TPU-side runner; run root `/home/ext_barongracias_gmail_com/tpu-runs/part-i/R1-grpo-full-s0-20260608_165231`; launched `2026-06-08T16:52:37.269465Z`; W&B URL `https://wandb.ai/barongracias-university-of-cambridge/agentic-ai-coursework/runs/R1-grpo-full-s0`.
- Configuration: GRPO, K inferred default 2, seed 0, max steps 3364, save interval 500, max checkpoints kept default 4, dataset `tfds`, greedy eval.
- Runtime/platform: Google TPU VM per notes; train log contains `Training finished.`; final log scan noted no fatal training marker. Wall-clock duration not extracted locally.
- Artefacts: metadata `ckpts/run_metadata.json`; train log `logs/train.log`; TensorBoard event `tensorboard/events.out.tfevents.1780937570.t1v-n-0339f27d-w-0`; scalar summary `eval/r1_tensorboard_scalar_summary.txt`; retained checkpoints `ckpts/actor/2000`, `2500`, `3000`, `3364`; eval CSVs under `eval/`; eval logs under `logs/`.
- Metrics:
  - base `--no-restore`: exact 31/64 (48.44%), partial 31/64 (48.44%), format 1/64 (1.56%), empty 0/64, restored step blank.
  - step 2000: exact 24/64 (37.50%), partial 26/64 (40.62%), format 28/64 (43.75%), empty 0/64, restore confirmed.
  - step 2500: exact 11/64 (17.19%), partial 11/64 (17.19%), format 23/64 (35.94%), empty 0/64, restore confirmed.
  - step 3000: exact 8/64 (12.50%), partial 8/64 (12.50%), format 16/64 (25.00%), empty 0/64, restore confirmed.
  - step 3364: exact 12/64 (18.75%), partial 12/64 (18.75%), format 23/64 (35.94%), empty 0/64, restore confirmed.
- Caveats: final checkpoint underperforms base; best retained checkpoint still under base. Initial eval log includes TFDS/protobuf tracebacks before successful rerun. Eval manifest field empty, so deterministic manifest was not used. W&B step-order warnings noted; TensorBoard/exported scalars are safer.

### R3-rloo-full-s0-20260608_230810

- Identity: Baron TPU-side runner; run root `/home/ext_barongracias_gmail_com/tpu-runs/part-i/R3-rloo-full-s0-20260608_230810`; launched `2026-06-08T23:08:40.142902Z`; W&B URL `https://wandb.ai/barongracias-university-of-cambridge/agentic-ai-coursework/runs/R3-rloo-full-s0-20260608_230810`.
- Configuration: RLOO, K inferred default 2, seed 0, max steps 3364, save interval 500, dataset `tfds`, greedy eval.
- Runtime/platform: Google TPU VM per notes; train log contains `Training finished.`.
- Artefacts: metadata `ckpts/run_metadata.json`; train log `logs/train.log`; TensorBoard event `tensorboard/events.out.tfevents.1780960133.t1v-n-0339f27d-w-0`; scalar summary `eval/r3_tensorboard_scalar_summary.txt`; retained checkpoints `ckpts/actor/2000`, `2500`, `3000`, `3364`; eval summary `eval/r3_eval_summary.txt`; eval CSVs/logs.
- Metrics:
  - step 2000: exact 1/64 (1.56%), partial 1/64 (1.56%), format 10/64 (15.62%), empty 52/64, restore confirmed.
  - step 2500: exact 6/64 (9.38%), partial 7/64 (10.94%), format 16/64 (25.00%), empty 45/64, restore confirmed.
  - step 3000: exact 1/64 (1.56%), partial 2/64 (3.12%), format 4/64 (6.25%), empty 60/64, restore confirmed.
  - step 3364: exact 1/64 (1.56%), partial 1/64 (1.56%), format 5/64 (7.81%), empty 58/64, restore confirmed.
- Caveats: no same-run base CSV found locally; compared in notes to R1 base 31/64. This is report-usable mainly as negative RLOO K=2 evidence. Eval manifest field empty. TFDS fresh-cache warnings did not stop eval.

### D3-grpo-k8-debug-20260609_094815

- Identity: Baron TPU-side K=8 debug; run root `/home/ext_barongracias_gmail_com/tpu-runs/part-i/D3-grpo-k8-debug-20260609_094815`; launched `2026-06-09T09:48:20.745804Z`; W&B URL `https://wandb.ai/barongracias-university-of-cambridge/agentic-ai-coursework/runs/D3-grpo-k8-debug-20260609_094815`.
- Configuration: GRPO, K=8 from notes/logs, seed 0, max steps 50, save interval 50, dataset `tfds`, greedy eval.
- Runtime/platform: Google TPU VM per notes; train log contains `Training finished.`.
- Artefacts: metadata `ckpts/run_metadata.json`; train log `logs/train.log`; checkpoint `ckpts/actor/50`; eval CSV `eval/d3_grpo_k8_step50_greedy.csv`; eval log `logs/eval_d3_grpo_k8_step50_greedy.log`; patch snapshots `patches/uncommitted_diff_before_d3.patch` and `patches/git_status_before_d3.txt`.
- Metrics: step 50, exact 31/64 (48.44%), partial 32/64 (50.00%), format 8/64 (12.50%), empty 0/64, restore confirmed.
- Caveats: metadata did not record `num_generations`; K=8 is supported by run notes and train log. TPU-side patch was uncommitted at launch. Debug only.

### D4-grpo-k8-medium-debug-20260609_100945

- Identity: Baron TPU-side K=8 medium diagnostic; run root `/home/ext_barongracias_gmail_com/tpu-runs/part-i/D4-grpo-k8-medium-debug-20260609_100945`; launched `2026-06-09T10:09:51.322394Z`; W&B URL `https://wandb.ai/barongracias-university-of-cambridge/agentic-ai-coursework/runs/D4-grpo-k8-medium-debug-20260609_100945`.
- Configuration: GRPO, K=8, seed 0, max steps 500, save interval 100, max checkpoints kept default 4, dataset `tfds`, greedy eval.
- Runtime/platform: Google TPU VM per notes; train log contains `Training finished.`.
- Artefacts: metadata `ckpts/run_metadata.json`; train log `logs/train.log`; TensorBoard event `tensorboard/events.out.tfevents.1780999805.t1v-n-0339f27d-w-0`; retained checkpoints `ckpts/actor/200`, `300`, `400`, `500`; eval summary `eval/d4_eval_summary.txt`; eval CSVs/logs; patch snapshots under `patches/`.
- Metrics:
  - step 100: missing/pruned, not evaluated locally.
  - step 200: exact 26/64 (40.62%), partial 28/64 (43.75%), format 55/64 (85.94%), empty 0/64, restore confirmed.
  - step 300: exact 32/64 (50.00%), partial 34/64 (53.12%), format 51/64 (79.69%), empty 0/64, restore confirmed.
  - step 400: exact 29/64 (45.31%), partial 29/64 (45.31%), format 55/64 (85.94%), empty 0/64, restore confirmed.
  - step 500: exact 34/64 (53.12%), partial 35/64 (54.69%), format 54/64 (84.38%), empty 0/64, restore confirmed.
- Caveats: diagnostic horizon only; step 100 pruned by `MAX_TO_KEEP=4`; no same-run base CSV found locally; eval manifest field empty.

### R5-grpo-k8-full-s0-20260609_114832

- Identity: Baron full K=8 GRPO run; run root `/home/ext_barongracias_gmail_com/tpu-runs/part-i/R5-grpo-k8-full-s0-20260609_114832`; launched `2026-06-09T11:48:38.217455Z`; W&B URL `https://wandb.ai/barongracias-university-of-cambridge/agentic-ai-coursework/runs/R5-grpo-k8-full-s0-20260609_114832`.
- Configuration: GRPO, `NUM_GENERATIONS=8`, seed 0, max steps 3364, save interval 250, `MAX_TO_KEEP=20`, dataset `tfds`, greedy eval. Reward function code is the existing format/numeric reward mix; local audit warns it over-rewards well-formatted wrong answers.
- Runtime/platform: Google TPU VM per notes; train log contains `Training finished.`; no fatal/OOM/traceback found in notes scan.
- Artefacts: metadata `ckpts/run_metadata.json`; train log `logs/train.log`; TensorBoard event `tensorboard/events.out.tfevents.1781005731.t1v-n-0339f27d-w-0`; retained checkpoint dirs `ckpts/actor/250`, `500`, `750`, `1000`, `1250`, `1500`, `1750`, `2000`, `2250`, `2500`, `2750`, `3000`, `3250`, `3364` plus step `1`; eval summary `eval/r5_all_retained_eval_summary.txt`; eval CSVs/logs for every retained checkpoint.
- Metrics:
  - step 250: exact 29/64 (45.31%), partial 30/64 (46.88%), format 58/64 (90.62%), empty 0/64.
  - step 500: exact 34/64 (53.12%), partial 35/64 (54.69%), format 54/64 (84.38%), empty 0/64.
  - step 750: exact 31/64 (48.44%), partial 34/64 (53.12%), format 60/64 (93.75%), empty 0/64.
  - step 1000: exact 32/64 (50.00%), partial 34/64 (53.12%), format 60/64 (93.75%), empty 0/64.
  - step 1250: exact 30/64 (46.88%), partial 35/64 (54.69%), format 59/64 (92.19%), empty 0/64.
  - step 1500: exact 29/64 (45.31%), partial 31/64 (48.44%), format 57/64 (89.06%), empty 0/64.
  - step 1750: exact 32/64 (50.00%), partial 34/64 (53.12%), format 56/64 (87.50%), empty 0/64.
  - step 2000: exact 31/64 (48.44%), partial 33/64 (51.56%), format 57/64 (89.06%), empty 0/64.
  - step 2250: exact 30/64 (46.88%), partial 31/64 (48.44%), format 58/64 (90.62%), empty 0/64.
  - step 2500: exact 33/64 (51.56%), partial 34/64 (53.12%), format 58/64 (90.62%), empty 0/64.
  - step 2750: exact 31/64 (48.44%), partial 32/64 (50.00%), format 63/64 (98.44%), empty 0/64.
  - step 3000: exact 31/64 (48.44%), partial 32/64 (50.00%), format 56/64 (87.50%), empty 0/64.
  - step 3250: exact 35/64 (54.69%), partial 36/64 (56.25%), format 55/64 (85.94%), empty 0/64.
  - step 3364: exact 32/64 (50.00%), partial 33/64 (51.56%), format 56/64 (87.50%), empty 0/64.
  - Restore was confirmed in each eval log. No R5 same-run base CSV was found locally; notes compare against R1 base 31/64.
- Caveats: best checkpoint selection matters; final is weaker than best retained. Eval manifest field is empty despite later deterministic manifest code. No R5 TensorBoard scalar summary export was found locally, only the event path. Report should avoid strong improvement claims until paired/bootstrap uncertainty is computed.

### R6-grpo-k2-baseline-s0-20260609_212708

- Identity: local deterministic K=2 baseline attempt; run root `/home/ext_barongracias_gmail_com/tpu-runs/part-i/R6-grpo-k2-baseline-s0-20260609_212708`; launched `2026-06-09T21:27:14.228577Z`; W&B run id `p6sx09d8`, URL in train log.
- Configuration: GRPO, K=2, seed 0, max steps 3364, save interval 250, `MAX_TO_KEEP=20`, `EVAL_MANIFEST=/home/ext_barongracias_gmail_com/tpu-runs/part-i/manifests/gsm8k_test_seed0_n64.jsonl`, dataset `tfds`.
- Runtime/platform: train log contains `Training finished.`; earlier root `/home/ext_barongracias_gmail_com/tpu-runs/part-i/R6-grpo-k2-baseline-s0-20260609_212334` failed with W&B permission denied.
- Artefacts: metadata `ckpts/run_metadata.json`, train log, TensorBoard event. No local eval CSVs or eval summary were found.
- Metrics: not found locally.
- Caveats: manifest directory exists but is empty locally, so the metadata manifest path is not currently backed by a local JSONL. Not report-usable until eval CSVs and the manifest are provided.

## Hard-Question Mining

No local Baron hard-question mining run was found.

- mining method: not found locally for Baron. Notes discuss a possible future GSM8K-train hard subset mined by frozen-base inference, but no completed local mining artefact was found.
- source baseline: proposed frozen base model in notes only; no local Baron mining output.
- manifest path: no Baron hard-training manifest found. `/home/ext_barongracias_gmail_com/tpu-runs/part-i/manifests` exists but is empty locally.
- number of questions: not found locally.
- criteria for hard: proposed in notes as greedy wrong and/or sampled pass/fail counts, but not executed locally.
- branch/commit containing code: current `scripts/data.py`/`scripts/evaluate.py` support eval manifests at `57c6409`; Fred's branch reportedly has `scripts/mine_hard_examples.py`, but that is outside Baron local evidence.
- runs using the manifest: none found locally. R6 references a deterministic eval manifest path, but the JSONL is not present locally and no eval CSVs were found. No local run uses a hard-question training manifest.

## Missing Evidence Checklist

- R5 paired/bootstrap confidence intervals from per-prompt CSVs.
- R5 TensorBoard scalar summary export for reward, KL, and completion length curves; only the event file path was found locally.
- Same-run base eval CSVs for R3, D3, D4, and R5 on exactly the same eval sample; R1 base exists.
- Deterministic eval manifest JSONL `/home/ext_barongracias_gmail_com/tpu-runs/part-i/manifests/gsm8k_test_seed0_n64.jsonl`; local manifest directory is empty.
- Wall-clock start/end durations for report table, if W&B runtime is not queried.
- Explicit dirty-worktree snapshots for D1/D2/R1/R3/R5 launch states; D3/D4 patch/status snapshots exist, but full dirty state for other launches is not found locally.
- Harvey K=8/RLOO artefacts mentioned in `origin/baron` team action note, if those runs are to be included.
- Confirmation whether `baron_k8` was intended only as a branch alias or actually launched any personal-TPU run; no distinct local branch-run proof was found.

## Report-Ready Claims

- R1 GRPO K=2 completed to step 3364 and restored checkpoints, but its best retained checkpoint was 24/64 exact and final was 12/64, below the local base reference of 31/64.
- R3 RLOO K=2 completed to step 3364 but collapsed badly in greedy eval, with final 1/64 exact and 58/64 empty responses.
- D4 K=8 medium diagnostic completed to step 500 and reached 34/64 exact with no empty responses, but it is diagnostic rather than final evidence.
- R5 GRPO K=8 completed to step 3364 with retained checkpoint evals every 250 steps; best retained step 3250 reached 35/64 exact, final step 3364 reached 32/64, and all evaluated R5 checkpoints had 0/64 empty responses.
- W&B URLs exist in local logs for the Baron runs, but W&B data was not queried during this audit.

## Not Yet Report-Ready

- A strong claim that K=8 improves over base: R5 best is 35/64 versus base 31/64, but uncertainty and same-run base comparability still need to be computed.
- Any claim based on `baron_k8` as a separate branch-run: no distinct local launch evidence was found.
- Any hard-question mining/curriculum claim for Baron: no local hard manifest or training run was found.
- R6 deterministic baseline claims: training completed, but eval artefacts and the referenced manifest are not found locally.
- Harvey K=8/RLOO claims: only action-note references were found in this fork; artefacts are pending/not found locally.

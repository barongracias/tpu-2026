# Fred verification note - 2026-06-13

Scope: checked `context/context.md`, `context/report_notes.md`, `context/experiments.md`, local `context/team_actions_fred_harvey_20260612.md`, and relevant local `../tpu-2026` notes/scripts/branches. I could not find `experiments/team_evidence/` or `experiments/team_evidence/evidence_fred_20260612.md` in this `AgentsCW` checkout.

## 1. Confirmed rows

- Fred R7 hard-medium GRPO: confirmed as GRPO, K=2, seed 0, run root `/home/fredlawrence/tpu-runs/part-i/R7-grpo-k2-hardmedium-20260610_102859`, W&B run id `3v8bx0iz`, `TRAIN_MANIFEST=/home/fredlawrence/tpu-runs/part-i/manifests/gsm8k_train_base_hard_medium.jsonl`, `SAVE_INTERVAL_STEPS=250`, `MAX_TO_KEEP=20`, final checkpoint `ckpts/actor/3452`, and operational completion. Local notes say the run collapsed to immediate-EOS/empty completions: first zero-length train completion at step `457`, persistent zero-length run from step `518`, final train/eval mean length `0`, and final train `actor/train/grad_norm=0`, `actor/train/kl=0`, `actor/train/loss=0`.
- Fred R7 eval caveat: confirmed that `/home/fredlawrence/tpu-runs/part-i/R7-grpo-k2-hardmedium-20260610_102859/eval/r7_best_greedy.csv` is base/no-restore, not trained R7. It has `31/64` exact, `31/64` partial, `1/64` format, `0/64` empty, and must not be reported as trained accuracy.
- Fred R8 hard-medium GRPO: confirmed as GRPO, K=8, seed 0, run root `/home/fredlawrence/tpu-runs/part-i/R8-grpo-k8-hardmedium-20260610_131135`, W&B run id `xnn69ylg` from the team action note, same hard-medium train manifest, and final trained eval at step `3452`.
- Fred R8 final trained metrics: confirmed from local `../tpu-2026/agents/context.md` and `agents/plan.md`: `/home/fredlawrence/tpu-runs/part-i/R8-grpo-k8-hardmedium-20260610_131135/eval/r8_step3452_greedy.csv`, restored `ckpts/actor` at step `3452`, exact `30/64` (`46.88%`), partial `32/64` (`50.00%`), format `60/64` (`93.75%`), empty `0/64`, mean response length about `180.5` words.
- Fred R8 base/no-restore eval caveat: confirmed that `/home/fredlawrence/tpu-runs/part-i/R8-grpo-k8-hardmedium-20260610_131135/eval/best_greedy.csv` is byte-identical to the R7 base/no-restore CSV, has empty `restored_step`, and should not be reported as trained R8.

## 2. Corrections needed

- The register should not list Fred R8 as clean `57c6409` provenance without a caveat. Commit `57c6409add0d81bbdb32ca7f4b3e176b4e044068` does not contain the `TRAIN_MANIFEST`/manifest-training support needed by `scripts/run_r8_grpo_k8_hardmedium.sh`. The relevant code was later committed on `fred`/`origin/fred` as `a93ea9f387b8c5083607263daf157929482c50cb`. Since R7/R8 ran on 2026-06-10 and `a93ea9f` was committed on 2026-06-11, safest wording is: based on `57c6409` plus dirty/untracked hard-mining and manifest-training changes later committed as `a93ea9f`; exact launch commit not proven locally.
- Fred R8 `Best retained exact` should not imply a retained-checkpoint sweep. Local evidence only confirms the final trained checkpoint eval at step `3452`, exact `30/64`. Earlier retained checkpoints such as 500/1500/2500 remain unevaluated or not locally evidenced.
- Fred hard-mining counts in the register (`1846` hard, `1990` medium, `3637` easy) are not locally verifiable from this checkout. They may be true, but I could not find the mining summary, CSV, or JSONL manifests locally. Mark those counts uncertain until the mining artefacts are supplied.
- Fred R9 stage-1/staged-reward row is not locally verified. I found no local `R9`, `stage 1`, `format-heavy`, or staged-reward evidence in `AgentsCW` or `../tpu-2026` notes/scripts. Keep it context-only or downgrade to unverified unless the evidence file/run artefacts are provided.
- The register's base comparison for R8 is directionally valid only for the held-out `gsm8k_test_seed0_n64.jsonl` eval, using the base/no-restore `31/64` CSV. The same-manifest raw CSV exists only as a TPU path in local notes, not in this checkout.

## 3. Missing artefacts

- `experiments/team_evidence/evidence_fred_20260612.md` is absent from this `AgentsCW` checkout.
- Hard-mining artefacts are absent locally: `/home/fredlawrence/tpu-runs/part-i/manifests/gsm8k_train_base_hard_medium.jsonl`, `/home/fredlawrence/tpu-runs/part-i/manifests/gsm8k_test_seed0_n64.jsonl`, and the mining summary/CSV/completions supporting the easy/medium/hard counts.
- Raw R7/R8 artefacts are absent locally: `ckpts/run_metadata.json`, train logs, TensorBoard scalar exports/event paths, final/retained eval CSVs, and eval logs. Local notes point to TPU paths, but the files are not in the submission checkout.
- Full W&B URLs are missing from the register for Fred R7/R8. Local evidence gives run ids `3v8bx0iz` and `xnn69ylg`, but not full URLs in `context/experiments.md`.
- R8 retained-checkpoint evals are missing; only final step `3452` is locally evidenced.
- R7 trained-checkpoint eval is missing; the likely salvage eval should restore `--ckpt-dir /home/fredlawrence/tpu-runs/part-i/R7-grpo-k2-hardmedium-20260610_102859/ckpts --step 250`.

## 4. Report-ready claims from my runs

- R8 can be used cautiously as a negative/stability result if the report cites it as final-checkpoint-only: K=8 GRPO on the hard/medium manifest avoided the R7 empty-output collapse at final step `3452`, with exact `30/64` versus same held-out base/no-restore `31/64`, and format `60/64`.
- R7 can be used as a training-collapse diagnostic, not as an accuracy row, unless a trained checkpoint eval is recovered. The supported claim is that K=2 GRPO on the hard/medium manifest collapsed to empty/immediate-EOS completions despite operational training completion.
- Hard-mining is useful as method/provenance context, but the manifest counts and exact selection set are not report-ready from this checkout until the manifests/mining summary are included.

## 5. Claims to avoid or qualify

- Do not claim a trained R7 exact-accuracy result from `r7_best_greedy.csv`; it is base/no-restore.
- Do not claim R8 improved exact accuracy over base; final trained R8 is `30/64`, below the base/no-restore `31/64` on the held-out eval.
- Do not call R8's `30/64` a best checkpoint. It is the only confirmed trained checkpoint eval locally.
- Do not present the Fred rows as clean `57c6409` runs without noting dirty/untracked manifest-training changes later committed as `a93ea9f`.
- Do not use R9 in a report table unless its evidence is supplied; I could not verify it locally.
- Do not present hard-mining counts as audited facts until the JSONLs/summary are copied into the repo or otherwise made accessible.

## 6. Recommended next action

Update `context/experiments.md` Fred rows to add the provenance caveat above, change R8's `Best retained exact` wording to final-only, downgrade hard-mining counts to uncertain until artefacts are present, and downgrade/remove the R9 row unless evidence is provided. Highest-value artefacts to add are the Fred evidence file, hard/medium manifest JSONLs, R8 `r8_step3452_greedy.csv`, R8 base/no-restore CSV, R7 TensorBoard scalar export, R7/R8 `run_metadata.json`, and a properly restored R7 step-250 eval if the report wants an accuracy row.

## 7. Local resolution update - 2026-06-13

Resolved in this checkout:

- The Fred evidence/register note now marks R7/R8 as `57c6409` plus
  dirty/untracked manifest-training changes later committed as `a93ea9f`,
  rather than clean `57c6409` provenance.
- R8 is now described as final-checkpoint-only evidence at step `3452`
  (`30/64` exact), not as a best-retained-checkpoint result.
- Hard-mining artefacts are now present under `experiments/evidence/`:
  `hard_mining_base_gsm8k_train_20260609_220117/summary.md`,
  `hard_mining_base_gsm8k_train_20260609_220117/run_metadata.json`, and
  the two manifest JSONLs under `experiments/evidence/manifests/`.
- R7/R8 lightweight artefacts are present under `experiments/evidence/`,
  including R7/R8 `run_metadata.json`, R8 `r8_step3452_greedy.csv`, and the
  base/no-restore CSVs.
- TensorBoard scalar exports and JSON summaries are now present for R7, R8,
  and R9 under each run's `experiments/evidence/<run-id>/` directory.
- R9 is no longer context-only in this checkout. The per-prompt eval CSVs,
  stage metadata, launch commit, and summary files have been copied to
  `experiments/evidence/R9-rloo-k8-format-answer-full-20260612_131553/`, with
  a local README summarising report-safe claims.

Still pending if needed for stronger report coverage:

- R7 trained-checkpoint salvage eval, likely step `250`.
- R8 retained-checkpoint eval sweep, for example steps `500`, `1500`, and
  `2500`, if best-checkpoint selection is required.
- Exact TPU type/zone and direct W&B runtime/status verification.

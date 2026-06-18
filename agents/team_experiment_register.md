# Team Experiment Register - 2026-06-13

This register collates the submitted evidence files under `experiments/team_evidence/`. It is a provenance and triage document for report analysis, not a final statistical analysis. Claims below should be treated as report-ready only where the row has a restored checkpoint evaluation, a clear branch/commit, and enough configuration detail to make the comparison reproducible.

## Evidence sources

| Source file | Owner | Notes |
|---|---:|---|
| `evidence_baron_20260612.md` | Baron | Internal runs from the original coursework branch and later deterministic work. Includes K=2 GRPO/RLOO, K=8 diagnostics, and R5 full K=8. |
| `evidence_fred_20260612.md` | Fred | Internal hard-question mining and hard/medium manifest runs. Includes R8 K=8 hard-medium and R9 staged reward attempt. |
| `evidence_harvey_20260612.md` | Harvey | Internal deterministic RLOO K=2 and K=8 runs. K=8 has strong retained-checkpoint result; K=2 collapses late. |
| `evidence_harvey_full_eval_20260614.md` | Harvey | Full GSM8K confirmation evals for deterministic RLOO K=2 and K=8 selected/final checkpoints, GRPO K=4/K=16, and the deterministic GRPO K=2 baseline reproduction, with paired bootstrap CIs. |
| `evidence_harvey_k4_full_eval_20260614.md` | Harvey | Full GSM8K confirmation eval for completed GRPO K=4 best-retained and final checkpoints, with paired bootstrap CIs. |
| `evidence_harvey_k16_full_eval_20260615.md` | Harvey | Full GSM8K confirmation eval for capped GRPO K=16 last-good checkpoint; training stopped early with `RESOURCE_EXHAUSTED`. |
| `baron_results_handoff_20260615.md` | Harvey | Compact response to Baron's 2026-06-15 handoff ask: GRPO K=4/K=16 status, provenance, scalar/eval artefact map, and optional RLOO full-test evidence. |
| `evidence_basia_20260612.md` | Basia/Barbara | External collaborator evidence for KL, length, empty-penalty, and G8+microbatch diagnostics. |
| `evidence_funmi_20260612` | Funmi | External collaborator evidence. File contains repeated pasted audits; the fuller middle section is treated as canonical for variant runs. |
| `evidence_rowan_20260612.md` | Rowan | External collaborator K sweep/reward-reweight evidence. Includes K=16 partial run and pending reward-reweight eval opportunity. |

## Executive summary

The strongest supported internal story is that K=2 training is unstable for this setup, while larger K materially improves stability and format adherence. The gains in exact GSM8K accuracy are modest on the 64-question greedy evaluation set: Baron R5 GRPO K=8 reaches 35/64 at its best retained checkpoint and 32/64 at the final checkpoint, against a base reference of 31/64. Harvey's deterministic RLOO K=8 reaches 35/64 at its best retained checkpoint on the 64-question screen, `722/1319` at that selected step on the full test set, and `742/1319` at the final checkpoint. The full-test final beats base by `+8.87` percentage points with a paired bootstrap 95% CI of `[+6.14, +11.68]` points. Harvey's GRPO K=4 run also beats base at both selected and final checkpoints, while the capped GRPO K=16 run reaches `703/1319` at its last-good checkpoint but stops early from TPU memory exhaustion. The deterministic GRPO K=2 baseline reproduction reaches `656/1319` at its selected checkpoint step `500`, but the paired CI versus base includes zero, and the final checkpoint falls to `218/1319`. In contrast, Baron/Harvey RLOO K=2 and Baron/Harvey GRPO K=2 degrade badly by the end of training, with RLOO K=2 showing severe empty-response collapse.

Fred's hard-question mining is valuable for method coverage and dataset provenance, but the completed K=8 hard-medium run does not beat the base exact score at final evaluation. External collaborators add useful coverage: KL, length, empty-penalty, LoRA, learning-rate, K=4/K=16, and reward-reweight attempts. These should be used as supporting context unless their evaluation protocol and provenance are reconciled with the internal runs.

## Status labels

| Label | Meaning |
|---|---|
| Report-ready | Restored checkpoint eval exists, core config is known, and the result can support a cautious claim. |
| Context only | Useful for diagnostics or broader coverage, but missing a clean comparison, final eval, or complete provenance. |
| Not usable | Training/eval missing, checkpoint missing, branch/config ambiguous, or run failed before producing analysable output. |
| Pending | Specific follow-up could make the run useful. |

## Internal team canonical runs

| Owner | Run | Branch / commit | Method | K | Seed | Eval status | Best retained exact | Final exact | Empty signal | Use |
|---|---|---|---|---:|---:|---|---:|---:|---:|---|
| Baron | D1 GRPO debug | `e3ebeef` | GRPO | 2 | 0 | step 50 restored | 30/64 | 30/64 | not flagged | Context only: auth/checkpoint/debug gate. |
| Baron | D2 RLOO debug | inferred `be631b2` | RLOO | 2 | 0 | step 50 restored | 30/64 | 30/64 | not flagged | Context only: RLOO accepted by code path. |
| Baron | R1 GRPO full | `4339ba8` | GRPO | 2 | 0 | retained checkpoints restored | 24/64 at step 2000 | 12/64 at 3364 | 0/64 final | Report-ready negative: K=2 over-optimises format and hurts exact accuracy. |
| Baron | R3 RLOO full | `99cb7f8` | RLOO | 2 | 0 | retained checkpoints restored | 6/64 at step 2500 | 1/64 at 3364 | 58/64 final | Report-ready negative: RLOO K=2 collapses to empty outputs late. |
| Baron | D3 GRPO K=8 debug | `99cb7f8` | GRPO | 8 | 0 | step 50 restored | 31/64 | 31/64 | 0/64 | Context only: K=8 debug passed without OOM/collapse. |
| Baron | D4 GRPO K=8 medium | `99cb7f8` | GRPO | 8 | 0 | retained checkpoints restored | 34/64 at step 500 | 34/64 at step 500 | 0/64 | Context/diagnostic: justified full K=8 run. |
| Baron | R5 GRPO K=8 full | `820fad6` | GRPO | 8 | 0 | all retained checkpoints restored | 35/64 at step 3250 | 32/64 at 3364 | 0/64 | Report-ready positive internal result, but improvement is modest and needs uncertainty. |
| Baron | R6 deterministic K=2 attempt | `57c6409` | GRPO | 2 | 0 | no local eval found | missing | missing | missing | Not usable until eval is found or rerun. |
| Fred | Hard mining base pass | `57c6409` plus mining script | base inference | n/a | 0 | mining complete | n/a | n/a | n/a | Report-ready for manifest provenance: 1846 hard, 1990 medium, 3637 easy from GSM8K train. |
| Fred | R7 hard-medium GRPO | `57c6409`, dirty/untracked mining changes | GRPO | 2 | 0 | trained eval not found | missing | missing | collapse seen in logs | Context only unless trained checkpoint eval is recovered. |
| Fred | R8 hard-medium GRPO | `57c6409` | GRPO | 8 | 0 | final checkpoint restored | 30/64 final | 30/64 at 3452 | 0/64 reported | Report-ready with caveat: hard-medium K=8 final is below base 31/64 exact. |
| Fred | R9 stage 1 format-heavy | `a93ea9f`, dirty reward/launcher changes | RLOO / staged reward | 8 | 0 | stage 1 eval restored | 30/64 at 500 | stage 2 failed | 0/64 reported | Context only: formatting gate passed; stage 2 failed before training. |
| Harvey | R7 RLOO K=2 deterministic | `71aab87` | RLOO | 2 | 0 | retained checkpoint evidence committed | 30/64 at step 500 | 1/64 at 3364 | 58/64 final | Report-ready for RLOO K=2 collapse, but original run root/logs not local. |
| Harvey | R7 RLOO K=8 deterministic | `e3d69a1` | RLOO | 8 | 0 | full base + step 2000 + step 3364 eval complete; retained checkpoints restored on 64-prompt screen | 722/1319 at step 2000 full eval | 742/1319 at 3364 full eval | 0/1319 at step 2000 and step 3364 full evals | Report-ready positive full-test RLOO K=8 result; final beats base by +8.87 pp. |
| Harvey | H GRPO K=4 full | `57c6409` | GRPO | 4 | 0 | full base + step 2750 + step 3364 eval complete; retained checkpoints restored on 64-prompt screen | 682/1319 at step 2750 full eval | 681/1319 at 3364 full eval | 0/1319 at step 2750 and step 3364 full evals | Report-ready positive full-test GRPO K=4 result; both selected and final beat base. |
| Harvey | H GRPO K=16 capped | `57c6409` | GRPO | 16 | 0 | full base + step 500 eval complete; training stopped early with `RESOURCE_EXHAUSTED` after scalar step 678 | 703/1319 at step 500 full eval | 703/1319 at step 500 last-good full eval | 0/1319 at step 500 | Report-ready as early-stopped/capped K=16 evidence only; not compute-matched to full-budget runs. |
| Harvey | Deterministic GRPO K=2 baseline | `7a77bce` | GRPO | 2 | 0 | full base + step 500 + step 3364 eval complete; retained checkpoints restored on 64-prompt screen | 656/1319 at step 500 full eval | 218/1319 at 3364 full eval | 0/1319 at step 500 and step 3364 full evals | Report-ready deterministic coursework baseline reproduction; selected checkpoint CI vs base includes 0, final checkpoint collapses. |
| Harvey | Older R6 RLOO variants | `8a0f7f2` | RLOO | 2/8 | unclear | no eval found | missing | missing | missing | Not usable. |

## External collaborator runs

| Owner | Run / W&B ID | Branch / commit | Method | K | Seed | Eval status | Best / final exact | Use |
|---|---|---|---|---:|---:|---|---|---|
| Basia | `8rmv0hgg` KL beta 1e-6 | `kl-control-bk` / `5e7d8f5` | GRPO, beta 1e-6 | 2 | 42 | final restored | final 33/64, base 33/64, format 0/64 | Report-ready external context: accuracy maintained but format collapses. |
| Basia | `oet2tfjd` KL beta 0.32 | `kl-control-bk` / `5e7d8f5`, dirty unknown | GRPO, beta 0.32 | 2 | 42 | final restored | final 0/64 | Report-ready external context: high-beta setting fails completely. |
| Basia | `jcp0b5cy` length penalty G2 | `reward-length-bk` / `f50d731`, dirty unknown | GRPO + length penalty | 2 | 42 | final restored | final 3/64 | Report-ready external context: length penalty alone does not rescue K=2. |
| Basia | `cyay16mj` length penalty G8 bs1 | `reward-length-on-g8-bk` / `89cde30` | GRPO + length penalty | 8 | 42 | step/final evals restored | final 32/64; control step 500 38/64 | Report-ready external context, but microbatch changed so comparison is not clean. |
| Basia | `dsi65u1z` empty penalty G2 | `empty-penalty-bk` / `2dac639`, dirty unknown | GRPO + empty penalty | 2 | 42 | step/final JSONLs committed | final 9/64; step 2000 4/64 | Context/report-ready with caveat: helps empty collapse but still weak exact accuracy. |
| Funmi | `jgs4c6kl` baseline seed42 | `baseline-fls` / `7e696c4` | GRPO | 2 | 42 | retained checkpoints restored | best 18/64 at 2000; final 2/64 | Report-ready external negative baseline for K=2 collapse. |
| Funmi | `hozux9t6` lr1e5 seed42 | `learning-rate-fls` / `99059ce7` | GRPO, LR variant, K provenance uncertain | probably 8 | 42 | retained checkpoints restored | best 20/64; final 19/64 | Context only until K/config provenance is settled. |
| Funmi | `aoz8dtkp` LoRA rank128 alpha128 | `lora-rank128-alpha128-fls` / `99059ce7` | GRPO + LoRA capacity | 8 | 42 | W&B final metrics only | restored exact missing | Not report-ready for accuracy. |
| Funmi | `v5cvlwkm` LoRA rank128 alpha128 G2 | `lora-rank128-alpha128-fls` / `762bb69` | GRPO + LoRA capacity | 2 | 42 | final restored | final 21/64, base 34/64 | Report-ready external context: capacity change does not prevent degradation. |
| Rowan | `dvsaxxyh` K=8 baseline | inferred `n-generations-8`, commit missing | GRPO | 8 | 42 | no local eval/checkpoint | formal eval missing | Context only; may duplicate `kosfzg6t`. |
| Rowan | `kosfzg6t` intended K=16 but actually K=8 | commit missing | GRPO | 8 | 42 | no local eval/checkpoint | formal eval missing | Context only; byte-identical to `dvsaxxyh`. |
| Rowan | `x4j7yhdp` K=4 | `n-generations-4` / `1db3d25` | GRPO | 4 | 42 | checkpoint deleted, no eval | missing | Not usable except W&B reward trend. |
| Rowan | `xqbl406c` K=16 | `n-generations-16` / `17c90d5` | GRPO | 16 | 42 | step 2500 restored before OOM | 36/64 at step 2500 | Partial report-ready external context: promising but failed/OOM before final. |
| Rowan | `smuzmoal` reward reweight | `reward-reweight` / `24583db` plus reward change `37847c1` | GRPO, reweighted reward | 8 | 42 | checkpoints retained; no accuracy eval yet | pending | Pending: high-priority eval candidate. |

## Best-supported comparisons

### Internal comparisons

| Comparison | Evidence | Current conclusion | Caveat |
|---|---|---|---|
| GRPO K=2 vs base | Baron R1 vs base greedy; Harvey deterministic baseline full-test reproduction | R1 best retained 24/64 and final 12/64, both below base 31/64. The deterministic full-test baseline selected step 500 reaches 656/1319 vs base 625/1319 but its CI includes 0; final step 3364 falls to 218/1319. Format improves at selected checkpoints but exact reasoning degrades late. | Single seed; best-checkpoint selection must be separated from final-checkpoint reporting. |
| RLOO K=2 vs base | Baron R3 and Harvey R7 K=2 | RLOO K=2 is unstable and collapses late; final exact 1/64 in both Baron/Harvey-style evidence. | Baron and Harvey runs are not identical branches; Harvey K=2 now has same-manifest full-test base comparison, while Baron remains 64-prompt evidence. |
| GRPO K=8 vs base | Baron R5 | K=8 is much healthier: best 35/64 and final 32/64 against base 31/64, no empty collapse. | Improvement is small; best-checkpoint selection needs to be described honestly. |
| GRPO K=4 vs base | Harvey H GRPO K=4 | K=4 is positive on the full test set: selected step 2750 reaches 682/1319 and final step 3364 reaches 681/1319, both above base 625/1319 with paired CIs excluding zero. | Same checkpoint-selection caveat for the selected checkpoint; this is a single seed. |
| GRPO K=16 vs base | Harvey H GRPO K=16 capped | Last-good step 500 reaches 703/1319, +5.91 pp over base with paired CI excluding zero. | Training stopped early from TPU memory exhaustion after scalar step 678; report as capped/OOM last-good only. |
| RLOO K=8 vs RLOO K=2 | Harvey R7 K=8 vs K=2 | Increasing K stabilises RLOO: K=8 reaches 722/1319 at the selected step 2000 and 742/1319 at final step 3364 with no empty collapse, while K=2 step 500 is 562/1319 and K=2 final has 1111/1319 empty responses. | The n=64 screen under-ranked the K=8 final checkpoint, so checkpoint-selection and final-checkpoint reporting should be separated. |
| Hard/medium mining | Fred mining + R8 | Mining produced a reproducible hard/medium manifest, but R8 final 30/64 does not beat the base 31/64. | R7 trained eval missing; R8 retained-checkpoint eval missing. |

### External coverage comparisons

| Theme | Evidence | What it adds | Caveat |
|---|---|---|---|
| KL sensitivity | Basia beta 1e-6 and 0.32 | Shows regularisation strength can preserve base-level exact with format collapse or destroy accuracy entirely. | Need curves and careful explanation of beta effects before using strongly. |
| Reward shaping | Basia length/empty penalty, Rowan reward reweight pending | Reward changes can change format/empty behaviour but do not automatically improve exact reasoning. | Some dirty status unknown; reward-reweight accuracy still pending. |
| K sweep | Baron/Harvey K=4/K=8/K=16, Rowan K=16/K=4 | Larger K improves stability relative to K=2 in the internal evidence. Harvey GRPO K=4 and capped K=16 now both have full-test evals above base; Harvey/Baron K=8 remains the strongest practical completed setting. | K=16 is early-stopped/OOM and not compute-matched to full-budget K=4/K=8. |
| LoRA/LR variants | Funmi | Larger LoRA capacity and LR changes do not solve the main issue by themselves. | LR K provenance uncertain; one LoRA run lacks restored exact eval. |

## Candidate report claims

The following are the safest claims to draft around, subject to final paired/bootstrap analysis:

1. In this implementation, K=2 training can make the model worse despite improving format-related reward. Baron R1 is the cleanest GRPO example: exact accuracy falls from base 31/64 to 24/64 at the best retained checkpoint and 12/64 at the final checkpoint.
2. The deterministic full-test GRPO K=2 baseline reproduction reaches `656/1319` at selected step `500`, but the paired CI versus base includes zero; final step `3364` falls to `218/1319` with a strongly negative paired CI.
3. RLOO with K=2 is particularly unstable in the collected runs. Baron R3 and Harvey K=2 both end near 1/64 exact with many empty outputs.
4. Increasing generations beyond K=2 is the strongest practical stabiliser observed internally. Harvey GRPO K=4 reaches `682/1319` at its selected checkpoint and `681/1319` at final, while Baron R5 GRPO K=8 has no empty outputs and reaches 35/64 exact at its best retained checkpoint. Harvey RLOO K=8 reaches the same best exact score on the 64-prompt screen, `722/1319` at the selected step `2000`, and `742/1319` at final step `3364` on the full test set.
5. The exact-accuracy improvement from K=8 is checkpoint-dependent but positive for both Harvey full-test checkpoints evaluated. Harvey RLOO K=8 step `2000` improves over the full-test base by `+7.35` points with paired 95% CI `[+4.62, +10.16]`; final step `3364` improves by `+8.87` points with paired 95% CI `[+6.14, +11.68]`. The report should distinguish the n=64 checkpoint-selection result from the full-test final result.
6. GRPO K=16 is promising but memory-limited in this setup: Harvey's capped run reaches `703/1319` at step `500`, but stops before the `2500` cap with `RESOURCE_EXHAUSTED`. Use it as feasibility and last-good evidence, not as a completed full-budget result.
7. Hard-question mining produced a clean hard/medium training subset, but Fred R8 does not yet show a final exact-accuracy improvement over base. It is useful as an attempted improvement and negative result.
8. External collaborator runs broaden the failure-mode map: KL strength, reward shaping, LoRA capacity, learning rate, and larger K all affect stability, but none is a simple guaranteed fix.

## Claims to avoid or qualify

- Do not generalise the Harvey/Baron K=8 improvements to every K=8 setup without matching branch, seed, checkpoint policy, and eval manifest. The effect remains checkpoint- and run-dependent.
- Do not compare external-team percentages as if every run used the same code, branch, seed, checkpoint policy, and eval manifest unless those are reconciled.
- Do not treat any K=16 evidence as a completed full run unless it actually reached the planned final step. Rowan K=16 failed/OOMed after step 2991, and Harvey GRPO K=16 stopped before its 2500-step cap with last-good checkpoint step 500.
- Do not use Fred R7 accuracy until a trained-checkpoint eval is recovered. The copied CSV appears to be base/mixup rather than R7-trained.
- Do not use Funmi `lr1e5_seed42` as a pure learning-rate ablation until the K/config provenance is resolved.
- Do not use `baron_k8` as a separate branch provenance claim unless the exact pushed branch/run link is verified.

## Missing artefacts and follow-ups

### High priority for the final report

1. Compute paired/bootstrap uncertainty for the main internal comparison: base vs Baron R5 step 3250 and step 3364, plus ideally Baron R1/R3. Harvey K8 step 2000 and step 3364 full-test uncertainty is complete.
2. Decide fixed-step vs best-checkpoint reporting. If using best checkpoint, state that checkpoint selection used the retained validation/eval sweep and report final checkpoint separately.
3. Export or collect the scalar curves needed for figures: reward components, KL, completion length, empty counts if available, and exact/format eval by checkpoint.
4. Build one clean results table for the report with only report-ready rows.
5. Verify whether external collaborator results can be cited in the team report and how to phrase them without implying ownership of their runs.

### Owner-specific gaps

| Owner | Gaps |
|---|---|
| Baron | Bootstrap CIs for R5; TensorBoard scalar export for R5; clarify `baron_k8` branch provenance; locate R6 eval or mark unusable. |
| Fred | Recover R7 trained eval if possible; eval R8 retained checkpoints; export TensorBoard scalars; record R9 stage-2 failure clearly. |
| Harvey | Original K=2 run root/logs if still available; decide whether to compute cross-run paired CIs among K=4/K=8/K=16 beyond the base comparisons. |
| Basia | Provide missing baseline `jgs4c6kl` and G8-rerun `4i8lcitv` JSONLs if used for bootstrap; resolve dirty statuses. |
| Funmi | Resolve K provenance for `lr1e5_seed42`; recover restored exact eval for `aoz8dtkp`; provide metadata/events if available. |
| Rowan | Evaluate `smuzmoal` retained checkpoints; verify if K=16 checkpoint/eval files still exist; avoid using K=4 for accuracy. |

## Recommended report table subset

For the main report, keep the table compact and favour the internally controlled evidence:

| Row | Why include |
|---|---|
| Base greedy reference, 31/64 or 625/1319 full-test | Anchor for all internal claims. |
| Deterministic GRPO K=2 baseline selected/final full-test | Coursework baseline reproduction: selected step 500 is inconclusive vs base, final step 3364 collapses. |
| Baron R1 GRPO K=2 best/final | Earlier 64-prompt evidence that format optimisation can hurt exact reasoning. |
| Baron R3 RLOO K=2 best/final | Shows severe RLOO K=2 empty collapse. |
| Baron R5 GRPO K=8 best/final | Main positive internal result. |
| Harvey GRPO K=4 selected/final full-test | Positive completed GRPO group-size result with paired CIs. |
| Harvey GRPO K=16 last-good full-test | Positive high-K feasibility result, clearly labelled early-stopped/OOM. |
| Harvey RLOO K=8 selected/full and final/full | Cross-check that K=8 also stabilises RLOO on the full test set. |
| Fred R8 hard-medium final | Negative result for hard-question mining. |
| Optional external rows: Basia KL/length, Rowan K=16 step2500, Funmi baseline | Use only if space allows and caveats are clear. |

## Figure candidates

1. Accuracy and format by checkpoint for Baron R1, Baron R5, Harvey RLOO K=2, and Harvey RLOO K=8.
2. Empty-response count by checkpoint for RLOO K=2 vs K=8.
3. Reward-component/KL traces for Baron R1 vs R5, if scalar exports are available.
4. Hard-mining composition bar: easy/medium/hard counts from Fred's base mining pass.
5. Optional external coverage heatmap: intervention vs final/best exact score, with caveat markers for failed or incomplete runs.

## Immediate next analysis task

Prepare a `report/data/` or `experiments/analysis/` table from the report-ready rows only, then compute paired/bootstrap confidence intervals for the base-vs-R5 comparison. Harvey's deterministic baseline, RLOO K=2-vs-K=8, and GRPO K=4/K=16 full-test comparisons now have same-manifest evidence and can be folded into that table. The current register is enough to choose which rows belong in that table; it is not yet a substitute for the final numeric analysis.

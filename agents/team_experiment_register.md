
Phase 1 — File inventory (what I actually opened)
Narrative / decision record (all read in full):

research_notes/log.md — append-only session log, newest-first (the spine)
research_notes/large_notes.md — the roadmap / notes-system rules
research_notes/phase_1.md, phase_2.md, phase_3.md, phase_4_plan.md, phase_4_data_audit.md — canonical phase specs
research_notes/forecast_gmm_plan.md, engineering_log.md, meeting_notes.md, report_plan.md, finalisation_checklist.md
README.md, report macros-results.tex, tables/, chapter list
Code I read to confirm notes-vs-reality (not just trusted):

faithfulness.py, gmm.py — read fully
method4_mrf.py (mean-shift fixed point), graph.py (k-NN), phase4_eval.py (λ* rule) — read the load-bearing functions
~/WeatherGenerator/scripts/gmm_inference.py — confirmed the --forecast-steps change exists
Artifacts I independently inspected (not via the log):

outputs/runs/phase_4_real/lambda_star.json → λ*=93.743, bracketed:true, clauses:[] ✓
outputs/runs/phase_4_forecast_softening/softening_by_lead.csv → reproduces the log's softening tables cell-for-cell ✓
outputs/data/phase_4_fc48_{6ep,14ep}_step{1..8}_2t.npz exist (the per-lead conversions are done)
Everything load-bearing that I checked matched. The rough edges are not code/notes disagreements — they're framing and scope gaps. Those are in the ledger.

Phase 2 — The story
Stage 0 — The problem as posed
What. The WeatherGenerator decoder emits, per spatial location, an independent Gaussian mixture (K=4) over a weather variable. You want one spatially coherent, high-likelihood field drawn from those marginals, post hoc, without retraining the decoder (README.md:6-9).

Theory. The decoder discarded the joint spatial covariance "for computational reasons" and kept only per-location marginals. So this is sampling a joint field whose only constraint is a set of independent 1-D mixtures plus an imposed coherence prior (large_notes.md:130, Working Decision #4).

Why. The alternative — retrain the head to emit dependence (the AtmoRep route) — was ruled out as architecturally closed. Stating that once stops the project relitigating "just retrain it" (large_notes.md:130).

Stage 1 — The toy, and a pivot in its design
What. An 8×8 single-channel toy: K=4 smooth quadratic mean surfaces, a seeded per-cell permutation of those means, and independent floored-Dirichlet mixture weights; σ fixed (homoscedastic) or radially varying (heteroscedastic) (phase_1.md:79-141).

Theory. Mixture-component labels are not spatially aligned — component k at one cell ≠ component k at its neighbour. So the sampler must couple on component values, never label index (constraint C5, phase_1.md:63-78).

Why / the pivot. Earlier toys used smooth-π ("regime-boundary") weights. A supervisor correction killed them: permuting smooth logits with the means secretly creates a smooth value-dominance field — a cross-location dependence the real decoder never emits, which would let a sampler cheat (phase_1.md:197-211, large_notes.md:76). The fix: keep spatial structure only in the mean surfaces, draw π independently. This also retired the "dominance-coherence" (C7) metric that depended on smooth-π.

Stage 2 — Methodology: pick a primary, but "carry both"
What. Built and scored several samplers on the homoscedastic toy:

Method 1 — regularised MAP: minimise NLL/N + λ·xᵀLx over an 8-neighbour graph Laplacian, warm-started at the mode field, Adam + restarts (phase_2.md:233-241).
Method 4 — extract each cell's GMM modes by 1-D mean-shift, then a value-space MRF (ICM) over the discrete modes (phase_2.md:292-302).
Baselines: iid draw, per-cell MAP, mixture-mean, variance-scaled, a*, and the smoothed-MAP "skeptic's baseline" (MAP then Laplacian blur).
Theory. Regularised MAP (soft, swept-λ) was chosen over the strict constrained Lagrangian because only the soft version yields a Pareto λ-curve and needs no KKT/projection machinery (phase_2.md:246-254). Method 4 rests on mean-shift mode-finding (Carreira-Perpiñán) and ICM on a Markov random field over values.

Why "carry both". Method 1 reaches lower roughness but does it by smearing values into the low-density valleys between modes — invisible on the (NLL, R̃) plane. A cross-method ΔNLL-to-best-mode diagnostic exposes it: at matched roughness Method 4 drifts ~6% of cells off-mode vs Method 1's ~14% and smoothed-MAP's 56% (phase_2.md:304-313). Neither dominates, so both were carried.

Two ablations that closed cleanly. (a) A TV/Huber penalty with an exact Ishikawa min-cut certificate: swapping the penalty doesn't fix smearing — the exact TV mode is clean down to R̃≈0.5 then collapses rather than smears, so the smear-vs-smoothness trade-off is structural, a property of the objective not the optimiser (phase_2.md:330-347). (b) Method 5 annealed-Langevin as a stochastic-search ablation of Method 1's objective — "nothing moved", Method 1's optimiser is validated (phase_2.md:315-328). Both were then cut from real-data scope.

Stage 3 — Evaluation: the honesty layer
What. A claim ladder and a metric set: rung 1 marginal faithfulness (NLL/N), rung 2 declared coherence (scale-free R̃ = S_edge/Var_V), rung 3 mode-respecting realism (ΔNLL smear tail + Method-1-vs-4), rung 4 physical realism (deferred) (large_notes.md:229-236).

Theory. The load-bearing realisation: in the GMM-only regime, spatial coherence is unfalsifiable — it's exactly the discarded covariance, so no GMM-derived quantity (variogram, spectrum, Moran's I) can validate it; they're descriptive-vs-failure-mode-brackets only (large_notes.md:222, phase_2.md:256-268). And the (NLL, roughness) plane is the optimiser's own axes, so it locates operating points but can't prove the method is good.

Why. This forces λ to be set by a declared rule (a physical correlation length / a matched baseline), never the Pareto "knee", and forces roughness to be reported scale-free, never as an absolute "lower is better". A separate idea — Sophie's global-GMM / quadrant-GMM validation — was the one way to get a real full-field reference on the toy, but it was deferred past deadline without a decision and CUT from Phase 4 (phase_3.md:107-115).

Stage 4 — Engineering interlude: getting a real GMM out of the model
What. Before any real data existed, the GMM training pipeline had to be made to run on Clariden. The standout fix: training crashed with -INF loss → NaN weights, because the loss counter guards used if loss > 0.0 — a pattern written for MSE. gmm_nll is a log-density; going negative is success, so the first negative batch stopped the counter and the normalisation blew up to -INF (engineering_log.md:86-127).

Theory. Log-likelihood is sign-unconstrained; MSE is non-negative. The bug is the single most instructive in the project (flagged as report material).

Why it matters downstream. With that and the env/SLURM/uenv fixes, gmm_era5_32ep_v3 trained to mini-epoch 31 and produced the first real artifact. Still OPEN: full multi-rank training dies on an NCCL watchdog collective-timeout; only inference is reliably green (engineering_log.md:287-326).

Stage 5 — Phase 4 data audit: the surprise
What. Audited the real extraction. Three expectations broke: the grid is O96 reduced Gaussian, not HEALPix L5; C=72 not ~73; and π is near-one-hot (median max-weight 0.965), the opposite of the soft-π (~0.35) the toy was calibrated to (phase_4_data_audit.md:18-25, 59-68). Practical bimodality is rare (1.5% at >2σ) and conspicuously token-blocky — partly a decoder artefact, on a single snapshot, from a 32-epoch debug-scale model.

Theory. Near-one-hot weights = a near-deterministic decoder; the mixture-mean field ≈ the MAP field. The blocky K_eff map aligned to token patches flags fidelity, not weather.

Why it reshaped the plan. Sampler value on this data was rescoped to (a) suppressing within-component σ noise everywhere and (b) mode choice on the ~1.5–14% multimodal subset. HEALPix machinery, physical units (no de-standardisation stats in the file), and global-GMM validation were all CUT (phase_4_data_audit.md:98-104, phase_4_plan.md:7-23).

Stage 6 — Phase 4 execution on real data
What. Ported the toy methodology to the O96 grid: a union-symmetrised k-NN sphere graph (|E|=162,406), sparse Laplacian, log-space NLL, a heteroscedastic mean-shift fix (v ← (Σ rμ/σ²)/(Σ r/σ²), needed because real cells have per-component σ), and a total λ roughness-matching rule* against smoothed-MAP. Result: λ*=93.74, bracketed, no fallback clause fired; Method 1 @ λ* beats smoothed-MAP at matched R̃ (−1.575 vs −1.333 NLL/N, ~2.8× fewer smeared cells); Method 4's β sweep is pinned by the near-one-hot unary gap — it never reaches the coherence target (log.md:117-131). I re-verified λ* and the all-clear directly from lambda_star.json.

Theory. The mean-shift fix is the exact stationary point of p'(v)=0 for per-component σ (reduces to the shared-σ update on the toys — confirmed at method4_mrf.py:141-144). λ* imports no new prior: the skeptic's own baseline anchors the coherence level.

Why the "carry both" verdict weakened here. On near-one-hot data, Method 4 has no room to operate — the second-best mode costs ~10 nats — so it survives only as the ΔNLL smear diagnostic for Method 1, not as a competing sampler. The toy's symmetric two-primary story does not carry to real data.

Stage 7 — The pivot that defines the report: "this is a nowcast"
What. Sophie's Slack reply triggered the key reframe: the canonical artifact is forecast-step 0 of a pure masked autoencoder — the model reconstructing the analysis state from partial tokens at the same time. Confirmed in config (forecast: {num_steps:0}) and in inference (preds.physical[0] hardcoded) (log.md:91-105).

Theory. Step-0 reconstruction is the lowest-uncertainty task in the pipeline — more deterministic than a real +1 step. So near-one-hot π is the expected signature, not a bug.

Why. This converts the entire Phase-4 result from "the sampler's main job" into the left-hand anchor of a two-regime curve — a sanity check in a controlled limit. To exercise the sampler's actual mode-choice value, you need genuinely multimodal GMMs, which means a forecast retrain.

Stage 8 — Forecast regime: building multimodality on purpose
What. Built gmm_forecast_config.yml + a --forecast-steps inference path (confirmed present in the WeatherGenerator repo), then extracted two checkpoints at leads +6h…+48h: 6-epoch gmm_fc48_v1 me5 (preview) and 14-epoch gmm_fc48_v2 me7 (warm-started continuation) (forecast_gmm_plan.md, log.md:29-90).

Theory & result. π softens monotonically with lead time — median max-π 0.95→0.77 (+48h), one-hot fraction 71%→9%, K_eff>1.5 reaching 90% (log.md:37-46, reproduced for me in softening_by_lead.csv). More training deepens the softening rather than sharpening back to one-hot, so it's not an undertraining artefact; and well-separated (>2σ) bimodality roughly doubles at long lead (2.9%→6.3%) — a genuinely new, training-emergent feature.

Why a lower bound. The warm start from the near-one-hot AE plus near-zero-residual forecast-engine init biases early checkpoints toward deterministic weights, so the 6-epoch numbers can only understate softening (forecast_gmm_plan.md:171-173).

Stage 9 — The reporting layer and the two-regime framing
What. Built the forecast-specific evaluation that didn't exist before: PIT, central coverage, Grimit closed-form mixture CRPS, fair-Gini ensemble CRPS, ΔCRPS do-no-harm, softening metrics, and an auto-emitter that writes macros-results.tex + three table fragments (log.md:3-27; code confirmed in faithfulness.py). Per-lead λ* re-selected at all 16 leads, all bracketed, M1 beats smoothed-MAP at every lead.

Theory. Two distinct notions kept rigorously apart (faithfulness.py:9-27): do-no-harm (iid ensemble ΔCRPS≈0, a construction identity for the stochastic baseline) vs marginal position (where a deterministic field's values sit in their own CDF). The report's non-claim #3 — no ensemble-calibration claim — lives in the code.

Why this framing. The narrative spine: the head's uncertainty is responsive, not decorative — ~0 at reconstruction, growing monotonically to +48h. The reconstruction regime is the controlled anchor; the forecast regime is where the sampler matters (report_plan.md:12-16).

Current state
The report is scaffolded, not written: chapters exist, auto-tables and 17/20 macros are filled (β trio em-dashed by design, confirmed in macros-results.tex), notebooks 04/05 execute clean — but the analytical prose is % TODO. The final artifact is fixed as the 14-epoch v2 me7. Several MUST items from the report plan are still open (see ledger flags).

Phase 3 — The decision ledger
Confidence = my confidence the decision is still correct/safe to put in the report.

#	Decision	Chosen	Alternatives rejected	Stated reason	Recorded	Conf.	Flag
1	Toy π design	Independent floored-Dirichlet π over permuted quadratic means	uniform-π; smooth-π regime-boundary; flat shared means	smooth-π smuggles a hidden value-dominance field the decoder doesn't emit	phase_1.md:197-211	High	—
2	Primary sampler	Method 1 regularised MAP (soft, swept-λ)	constrained Lagrangian/KKT	only soft version gives a Pareto curve; no projection machinery	phase_2.md:246-254	High	—
3	Keep Method 4 too	"Carry both" (M1 + M4)	pick one	M4 uniquely avoids smearing on the toy	phase_2.md:304-313	Med	Weakens on real data — M4's β pinned, so it's only a diagnostic, not a competitor. Don't present "carry both" as a Phase-4 result (log.md:23)
4	Cut Method 5 / exact-TV from real data	Cut	port at scale	toy ablations closed; "nothing moved"	phase_4_plan.md:16-18	High	—
5	Coherence is a declared prior, unfalsifiable in GMM-only regime	Report R̃ scale-free, λ from a declared rule	absolute roughness; Pareto knee	discarded covariance can't be re-validated from marginals	large_notes.md:222	High	—
6	λ* selection	Roughness-match to smoothed-MAP n10; λ*=93.74 bracketed	knee; hand-tune	skeptic's baseline anchors coherence, imports no new prior	lambda_star.json; phase4_eval.py:168	High	Verified in code + artifact (clauses==[])
7	Heteroscedastic mean-shift fix	v←(Σrμ/σ²)/(Σr/σ²)	shared-σ update	real cells violate shared-σ; exact p'=0	method4_mrf.py:141-144	High	Confirmed in code
8	Grid + graph	O96 + union k-NN (k=8) sphere graph	HEALPix face; mutual-kNN	audit confirmed O96; union mirrors toy E_8	phase_4_data_audit.md:70-75; graph.py:78	High	Confirmed in code
9	Phase-4 artifact = step-0 masked AE reconstruction	Treat as reconstruction-regime anchor, not the headline	call near-one-hot π a bug	config num_steps:0; lowest-uncertainty task	log.md:91-105	High	—
10	Build the forecast regime	+6h…+48h retrain via warm start	stay at step-0	step-0 doesn't exercise mode choice	forecast_gmm_plan.md	High	—
11	Final forecast artifact	14-epoch gmm_fc48_v2 me7	a future 32-epoch run	NCCL risk; softening already robust	log.md:29-33	Med	"Converged"/"14ep" labels overstate — log itself notes val-NLL flat but mode-structure still moving, and 14ep = 6+8 across two LR cycles, not one trajectory. Use cautious wording in the report
12	Faithfulness scope	iid ΔCRPS≈0 (do-no-harm); M1 = marginal position only	claim M1 calibration	non-claim #3	faithfulness.py:9-27	High	Easy to over-read: the ≈0 ΔCRPS is the iid baseline, NOT evidence for Method 1. M1 PIT-KS is 0.29–0.34. Keep these separate in prose
13	Stale headline number	+6h median max-π = 0.953 (not 0.965)	0.965	0.965 is the AE step-0 anchor; macros use 0.953	log.md:60; macros-results.tex	High	report_plan.md:151 §5.2 still says "0.965→0.80" — stale, conflicts with the corrected canonical value. Fix before quoting
14	Global/quadrant-GMM validation	Deferred past deadline	run it / reject it	9 Jun gate lapsed; CUT from Phase 4	phase_3.md:107-115	Med	Parked, never confirmed with Sophie; viva probe "where's the held-out reference?" partly unanswered
15	Physical units	Out of scope	de-standardise	no norm stats in the .pt file	phase_4_data_audit.md:36	Med	Report plan D9 still wants λ* in physical units (correlation length); that translation is planned, not done
Rough edges I'd personally re-confirm before they go in the report
Bootstrap CIs / error bars do not exist. The report plan lists them as a top viva probe and a MUST (report_plan.md:190-194), and Ch4 §4.2 status literally says "CIs MISSING". Every sampler comparison is currently single-point. This is the biggest unmet "must do".
Held-out evaluation protocol is unbuilt (report_plan.md:150 §5.1 "MISSING — design now"). The whole forecast regime runs on a single init datetime (2023-11-01T00:00) — no robustness across inits, no train/test split. RQ1's monotone-softening claim and every sampler number ride on one snapshot. The notes flag this honestly, but a reader/examiner will press it hard.
Debug-scale fidelity caveat contaminates RQ1. The token-blocky multimodality pattern means part of the multimodality census is decoder artefact, not weather (phase_4_data_audit.md:67-68). The descriptive "multimodality grows with lead" claim is partly measuring an undertrained 32-epoch model.
The report-plan's #1 blocker is now stale. "Launch the converged forecast run — everything else has a workaround; this doesn't" (report_plan.md:203) was effectively superseded by the decision to declare the 14-epoch v2 the final artifact (D11). The plan text hasn't caught up; don't let it drive remaining work.
"≥1σ multimodality grows" vs "strongly bimodal". The careful v1 wording ("not strongly bimodal") was partially overturned at v2 (>2σ doubles to 6.3%) but it's still a ~6% minority (log.md:52). The honest phrasing changed between checkpoints — make sure the report uses the v2-era wording, not the v1 caveat or an over-claim.

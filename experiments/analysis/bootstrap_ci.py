"""Bootstrap confidence intervals over per-prompt eval CSVs (shared team method).

This is the one bootstrap method the team should use so the "uncertainty" column
is comparable across everyone's runs. It consumes the per-prompt CSVs written by
``evaluate.py --output-csv`` (columns include ``prompt_id`` and ``correct``) and
produces:

  * per-model accuracy with a percentile bootstrap 95% CI (resample prompts with
    replacement), and
  * paired bootstrap CIs for model-vs-model differences, aligned by ``prompt_id``
    (the same resampled prompt indices are applied to both models, which is the
    correct paired test when both models answered the same prompts).

It is pure standard library (no numpy) and fully reproducible: every statistic is
seeded from ``--seed`` plus the task identity, so adding or removing an input does
not change another input's numbers.

Examples
--------
Per-model CIs for an entire checkpoint sweep::

    python experiments/analysis/bootstrap_ci.py \
        --glob 'experiments/evidence/R7-rloo-k2-*/eval/*step*greedy.csv'

Explicit labels, a baseline, and report-ready outputs::

    python experiments/analysis/bootstrap_ci.py \
        --csv base=eval/base_full.csv \
        --csv k2_best=eval/r7_rloo_k2_step500_greedy.csv \
        --csv k8_best=eval/r7_rloo_k8_step2000_greedy.csv \
        --baseline base \
        --pair k8_best k2_best \
        --resamples 10000 --seed 12345 \
        --output-json report_ci.json --output-csv report_ci.csv
"""
import argparse
import csv
import glob as globlib
import json
import os
import random


def percentile(sorted_xs, q):
    """Linear-interpolation percentile (matches numpy's default), q in [0, 100]."""
    if not sorted_xs:
        raise ValueError("empty sample")
    if len(sorted_xs) == 1:
        return sorted_xs[0]
    rank = (q / 100.0) * (len(sorted_xs) - 1)
    lo = int(rank)
    frac = rank - lo
    if lo + 1 >= len(sorted_xs):
        return sorted_xs[-1]
    return sorted_xs[lo] + frac * (sorted_xs[lo + 1] - sorted_xs[lo])


def task_rng(seed, *parts):
    """Deterministic RNG keyed on the base seed and the task identity."""
    return random.Random(f"{seed}|" + "|".join(str(p) for p in parts))


def load_csv(path, metric):
    """Return (label_meta, {prompt_id: value}) for the chosen 0/1 metric column."""
    by_id = {}
    manifest = None
    restored_step = None
    with open(path, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        if metric not in (reader.fieldnames or []):
            raise SystemExit(f"{path}: column '{metric}' not found. "
                             f"Available: {reader.fieldnames}")
        for row in reader:
            pid = int(row["prompt_id"])
            by_id[pid] = int(float(row[metric]))
            manifest = manifest or row.get("eval_manifest")
            restored_step = restored_step if restored_step is not None else row.get("restored_step")
    if not by_id:
        raise SystemExit(f"{path}: no rows.")
    return {"path": path, "manifest": manifest, "restored_step": restored_step}, by_id


def boot_single(values, resamples, rng):
    """Percentile bootstrap CI for the mean of a 0/1 list."""
    n = len(values)
    idx = range(n)
    stats = []
    for _ in range(resamples):
        pick = rng.choices(idx, k=n)
        stats.append(sum(values[i] for i in pick) / n)
    stats.sort()
    return percentile(stats, 2.5), percentile(stats, 97.5)


def boot_paired(a_vals, b_vals, resamples, rng):
    """Paired bootstrap of mean(a) - mean(b) over the SAME resampled indices."""
    n = len(a_vals)
    idx = range(n)
    diffs = []
    n_gt = 0
    for _ in range(resamples):
        pick = rng.choices(idx, k=n)
        da = sum(a_vals[i] for i in pick) / n
        db = sum(b_vals[i] for i in pick) / n
        d = da - db
        diffs.append(d)
        if d > 0:
            n_gt += 1
    diffs.sort()
    lo, hi = percentile(diffs, 2.5), percentile(diffs, 97.5)
    prob_gt0 = n_gt / resamples
    # Two-sided bootstrap p-value: 2 * the smaller tail mass at 0.
    n_ge0 = sum(1 for d in diffs if d >= 0)
    n_le0 = sum(1 for d in diffs if d <= 0)
    p_two = min(1.0, 2.0 * min(n_ge0, n_le0) / resamples)
    return lo, hi, prob_gt0, p_two


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", action="append", default=[], metavar="LABEL=PATH",
                    help="A labelled per-prompt CSV. Repeatable.")
    ap.add_argument("--glob", action="append", default=[], metavar="PATTERN",
                    help="Glob of CSVs; labels are derived from filenames. Repeatable.")
    ap.add_argument("--metric", default="correct",
                    choices=["correct", "partial_correct", "format_correct"],
                    help="Per-prompt 0/1 column to bootstrap (default: correct).")
    ap.add_argument("--baseline", default=None,
                    help="Label to compare every other model against (paired).")
    ap.add_argument("--pair", action="append", default=[], nargs=2, metavar=("A", "B"),
                    help="Compare A vs B (paired). Repeatable.")
    ap.add_argument("--resamples", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=12345)
    ap.add_argument("--output-json", default=None)
    ap.add_argument("--output-csv", default=None)
    args = ap.parse_args()

    # Collect labelled CSV paths.
    items = []  # (label, path)
    for spec in args.csv:
        if "=" not in spec:
            raise SystemExit(f"--csv expects LABEL=PATH, got: {spec}")
        label, path = spec.split("=", 1)
        items.append((label, path))
    for pattern in args.glob:
        for path in sorted(globlib.glob(pattern)):
            label = os.path.splitext(os.path.basename(path))[0]
            items.append((label, path))
    if not items:
        raise SystemExit("No inputs. Use --csv LABEL=PATH and/or --glob PATTERN.")

    models = {}      # label -> {"meta":..., "by_id": {...}}
    for label, path in items:
        if label in models:
            raise SystemExit(f"Duplicate label '{label}'.")
        meta, by_id = load_csv(path, args.metric)
        models[label] = {"meta": meta, "by_id": by_id}

    # Consistency warnings across inputs.
    manifests = {m["meta"]["manifest"] for m in models.values() if m["meta"]["manifest"]}
    if len(manifests) > 1:
        print("WARNING: inputs reference different eval_manifest values; paired "
              "comparisons are only valid on the shared prompt set:")
        for label, m in models.items():
            print(f"    {label}: {m['meta']['manifest']}")
    id_sets = {label: set(m["by_id"]) for label, m in models.items()}
    common = set.intersection(*id_sets.values()) if id_sets else set()

    # ---- Per-model accuracy + bootstrap CI ----
    per_model = []
    print(f"\nPer-model accuracy ({args.metric}), 95% bootstrap CI "
          f"[{args.resamples} resamples, seed {args.seed}]:")
    print(f"  {'label':<22} {'n':>5} {'acc':>8}   95% CI")
    for label in sorted(models):
        by_id = models[label]["by_id"]
        ordered_ids = sorted(by_id)
        values = [by_id[i] for i in ordered_ids]
        n = len(values)
        acc = sum(values) / n
        rng = task_rng(args.seed, "single", label, args.metric, args.resamples)
        lo, hi = boot_single(values, args.resamples, rng)
        per_model.append({
            "label": label, "n": n, "correct": sum(values), "acc": acc,
            "ci_low": lo, "ci_high": hi,
            "restored_step": models[label]["meta"]["restored_step"],
            "path": models[label]["meta"]["path"],
        })
        print(f"  {label:<22} {n:>5} {sum(values):>3}/{n:<3} {acc*100:6.2f}%  "
              f"[{lo*100:5.2f}%, {hi*100:5.2f}%]")

    # ---- Paired comparisons ----
    pairs = list(args.pair)
    if args.baseline:
        if args.baseline not in models:
            raise SystemExit(f"--baseline '{args.baseline}' not among inputs.")
        for label in sorted(models):
            if label != args.baseline:
                pairs.append([label, args.baseline])

    comparisons = []
    if pairs:
        print(f"\nPaired comparisons (A - B), 95% bootstrap CI on the {len(common)} "
              f"shared prompts:")
        print(f"  {'A vs B':<34} {'Δacc':>8}   95% CI{'':14} P(Δ>0)  p(2-sided)")
    for a, b in pairs:
        for lab in (a, b):
            if lab not in models:
                raise SystemExit(f"--pair references unknown label '{lab}'.")
        ids = sorted(set(models[a]["by_id"]) & set(models[b]["by_id"]))
        a_vals = [models[a]["by_id"][i] for i in ids]
        b_vals = [models[b]["by_id"][i] for i in ids]
        n = len(ids)
        diff = (sum(a_vals) - sum(b_vals)) / n
        rng = task_rng(args.seed, "pair", a, b, args.metric, args.resamples)
        lo, hi, prob_gt0, p_two = boot_paired(a_vals, b_vals, args.resamples, rng)
        sig = "" if (lo <= 0 <= hi) else "  *"
        comparisons.append({
            "a": a, "b": b, "n": n, "delta_acc": diff,
            "ci_low": lo, "ci_high": hi, "prob_gt0": prob_gt0, "p_two_sided": p_two,
            "ci_excludes_zero": not (lo <= 0 <= hi),
        })
        print(f"  {a + ' vs ' + b:<34} {diff*100:+7.2f}%  "
              f"[{lo*100:+6.2f}%, {hi*100:+6.2f}%]  {prob_gt0:6.3f}  {p_two:8.4f}{sig}")
    if pairs:
        print("  (* = 95% CI excludes 0)")

    out = {
        "metric": args.metric, "seed": args.seed, "resamples": args.resamples,
        "shared_prompts": len(common),
        "per_model": per_model, "comparisons": comparisons,
    }
    if args.output_json:
        os.makedirs(os.path.dirname(os.path.abspath(args.output_json)), exist_ok=True)
        with open(args.output_json, "w", encoding="utf-8") as fh:
            json.dump(out, fh, indent=2)
        print(f"\nWrote {args.output_json}")
    if args.output_csv:
        os.makedirs(os.path.dirname(os.path.abspath(args.output_csv)), exist_ok=True)
        with open(args.output_csv, "w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(["kind", "label_or_A", "B", "n", "acc_or_delta",
                        "ci_low", "ci_high", "prob_gt0", "p_two_sided"])
            for r in per_model:
                w.writerow(["model", r["label"], "", r["n"], f"{r['acc']:.6f}",
                            f"{r['ci_low']:.6f}", f"{r['ci_high']:.6f}", "", ""])
            for r in comparisons:
                w.writerow(["pair", r["a"], r["b"], r["n"], f"{r['delta_acc']:.6f}",
                            f"{r['ci_low']:.6f}", f"{r['ci_high']:.6f}",
                            f"{r['prob_gt0']:.6f}", f"{r['p_two_sided']:.6f}"])
        print(f"Wrote {args.output_csv}")


if __name__ == "__main__":
    main()

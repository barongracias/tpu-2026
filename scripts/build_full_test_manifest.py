"""Build the shared full GSM8K test manifest (all test prompts) for report eval.

This writes a JSONL manifest with the SAME schema and the SAME deterministic
order as the team's existing n=64 manifest, but containing the whole GSM8K test
split (1,319 prompts). Because the ordering reuses
``get_dataset(..., shuffle_seed=eval_seed)`` -- exactly what produced the n=64
manifest -- the existing 64-prompt manifest is, by construction, the first 64
rows of this file. That keeps ``prompt_id`` alignment between the cheap 64-prompt
screening evals and the full-test evals, which is what paired bootstrap needs.

Run ONCE on a TPU/dataset-capable VM (it downloads/loads GSM8K via TFDS), then
commit the resulting JSONL plus its ``.sha256`` sidecar so every teammate
evaluates against a byte-identical file:

    python scripts/build_full_test_manifest.py \
        --out experiments/manifests/gsm8k_test_seed0_full.jsonl \
        --eval-seed 0 \
        --verify-prefix \
          experiments/evidence/R7-rloo-k2-det-harvey-full-s0-20260611_102009/manifests/gsm8k_test_seed0_n64.jsonl

The printed SHA-256 is the value to share with the team (the analogue of the
n=64 manifest's 9aa1814...). Evaluate with:

    python scripts/evaluate.py --no-restore \
        --eval-manifest experiments/manifests/gsm8k_test_seed0_full.jsonl \
        --output-csv eval/base_full.csv
"""
import argparse
import hashlib
import json
import os

from config import DATA_SOURCE, EVAL_SEED, TEST_DATA_DIR
from data import as_text, get_dataset, manifest_row, question_hash


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True,
                    help="Output JSONL path for the full-test manifest.")
    ap.add_argument("--eval-seed", type=int, default=EVAL_SEED,
                    help="Shuffle seed for test ordering. Must match the seed used "
                         "for the existing n=64 manifest (default from config: "
                         f"{EVAL_SEED}).")
    ap.add_argument("--source", default=DATA_SOURCE, choices=["tfds", "kaggle"],
                    help="Dataset source (default from config).")
    ap.add_argument("--test-data-dir", default=TEST_DATA_DIR,
                    help="Where TFDS/Kaggle data is materialised (default from config).")
    ap.add_argument("--limit", type=int, default=None,
                    help="Optional cap on number of prompts (debug only; omit for full test).")
    ap.add_argument("--verify-prefix", default=None,
                    help="Existing shorter manifest (e.g. the n=64 file). Asserts its "
                         "rows are the prefix of this manifest by question_sha256.")
    args = ap.parse_args()

    ds = get_dataset(args.test_data_dir, "test", args.source, shuffle_seed=args.eval_seed)
    n_total = len(ds)
    n_write = min(n_total, args.limit) if args.limit else n_total
    print(f"GSM8K test split: {n_total} examples; writing {n_write} "
          f"(source={args.source}, eval_seed={args.eval_seed}).")

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    n_missing_answer = 0
    with open(args.out, "w", encoding="utf-8") as fh:
        for i in range(n_write):
            ex = ds[i]
            question = as_text(ex["question"])
            answer = ex["answer"]
            answer = as_text(answer) if answer is not None else ""
            if not answer:
                n_missing_answer += 1
            row = manifest_row(
                prompt_id=i + 1,
                question=question,
                answer=answer,
                source=args.source,
                split="test",
                eval_seed=args.eval_seed,
            )
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    if n_missing_answer:
        print(f"WARNING: {n_missing_answer} prompts had no parseable '#### <num>' answer.")

    digest = sha256_file(args.out)
    sidecar = args.out + ".sha256"
    with open(sidecar, "w", encoding="utf-8") as fh:
        fh.write(f"{digest}  {os.path.basename(args.out)}\n")
    print(f"Wrote {args.out}")
    print(f"SHA-256: {digest}")
    print(f"Sidecar: {sidecar}")

    if args.verify_prefix:
        with open(args.verify_prefix, encoding="utf-8") as fh:
            prefix_hashes = [json.loads(line)["question_sha256"]
                             for line in fh if line.strip()]
        with open(args.out, encoding="utf-8") as fh:
            full_rows = [json.loads(line) for line in fh if line.strip()]
        ok = True
        for k, want in enumerate(prefix_hashes):
            got = full_rows[k].get("question_sha256") or question_hash(full_rows[k]["question"])
            if got != want:
                ok = False
                print(f"PREFIX MISMATCH at row {k + 1}: {want} (existing) != {got} (full)")
                break
        if ok:
            print(f"PREFIX OK: first {len(prefix_hashes)} rows of {args.out} match "
                  f"{args.verify_prefix} by question_sha256. The n=64 set is a clean "
                  f"subset of the full manifest.")


if __name__ == "__main__":
    main()

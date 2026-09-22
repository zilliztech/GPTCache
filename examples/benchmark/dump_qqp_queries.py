"""Dump the QQP query subset used by benchmark_qqp.py to a JSON file.

Avoids re-loading HuggingFace `datasets` every time close_gaps.py runs.
Output schema:
    {
      "n_ingest": int,
      "n_tp": int,
      "n_fp": int,
      "db_questions": [str, ...],
      "tp_queries":   [str, ...],
      "fp_queries":   [str, ...]
    }
"""
import argparse
import json
import os
import sys
import time


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--scale", type=int, default=10000)
    p.add_argument("--n-tp", type=int, default=2000)
    p.add_argument("--n-fp", type=int, default=2000)
    p.add_argument("--out", default=os.path.join("..", "..", "bench_real_10k", "qqp_queries.json"))
    args = p.parse_args()

    print("Loading HF datasets...", flush=True)
    t0 = time.perf_counter()
    from datasets import load_dataset
    ds = load_dataset("glue", "qqp", split="train")
    print(f"  load_dataset: {time.perf_counter()-t0:.1f}s, len={len(ds)}", flush=True)

    t0 = time.perf_counter()
    dup = ds.filter(lambda x: x["label"] == 1)
    print(f"  filter dup: {time.perf_counter()-t0:.1f}s, len={len(dup)}", flush=True)

    t0 = time.perf_counter()
    non = ds.filter(lambda x: x["label"] == 0)
    print(f"  filter non: {time.perf_counter()-t0:.1f}s, len={len(non)}", flush=True)

    n_ingest = min(args.scale, len(dup))
    n_tp = min(args.n_tp, n_ingest)
    n_fp = min(args.n_fp, len(non))

    t0 = time.perf_counter()
    dup_pairs = list(dup.select(range(n_ingest)))
    print(f"  select+list dup_pairs ({n_ingest}): {time.perf_counter()-t0:.1f}s", flush=True)

    t0 = time.perf_counter()
    non_pairs = list(non.select(range(n_fp)))
    print(f"  select+list non_pairs ({n_fp}): {time.perf_counter()-t0:.1f}s", flush=True)

    db_questions = [p["question1"] for p in dup_pairs]
    tp_queries = [p["question2"] for p in dup_pairs[:n_tp]]
    fp_queries = [p["question2"] for p in non_pairs]

    out = {
        "n_ingest": n_ingest,
        "n_tp": n_tp,
        "n_fp": n_fp,
        "db_questions": db_questions,
        "tp_queries": tp_queries,
        "fp_queries": fp_queries,
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False)
    print(f"Wrote {args.out}  (db={len(db_questions)}, tp={len(tp_queries)}, fp={len(fp_queries)})", flush=True)


if __name__ == "__main__":
    main()

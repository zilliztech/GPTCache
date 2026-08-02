"""Additional corpora for the Semantic-ARC crossover study.

``prepare_data.py`` builds the two corpora the original brief named. This script
adds the ones needed to answer a different question -- *under what traffic does
ARC actually beat LRU* -- which needs (a) a second ground-truth cluster corpus,
so the crossover can be shown to be a property of the traffic rather than of
Quora, and (b) a real trace long enough to contain real drift.

===================  ======================================================
``stackexchange``    StackExchange duplicate-question titles, union-found
                     into paraphrase clusters. Ground truth, so false hit
                     rate is measurable. Technical register, much shorter and
                     more keyword-like than Quora -- a different similarity
                     geometry, which is the point.
``wildchat-long``    WildChat-1M across *all* 14 shards rather than 3, so the
                     trace spans the full release window instead of two
                     months. Real arrival order, real drift, no ground truth.
===================  ======================================================

Same embedding model, normalisation, and metadata contract as
``prepare_data.py`` -- these files drop into ``benchmarks/data/`` and are read
by ``traces.py`` exactly like the originals.

LMSYS-Chat-1M is still gated to this account (``GatedRepoError`` on download,
though the file listing is public), so the substitution documented in
``prepare_data.py`` still stands.

Usage
-----
::

    python benchmarks/prepare_data_ext.py                     # both
    python benchmarks/prepare_data_ext.py --only stackexchange
"""

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from prepare_data import (  # noqa: E402
    DATA_DIR,
    DIM,
    MODEL_NAME,
    UnionFind,
    embed,
    pick_device,
    sha256_file,
    write_trace,
)

SE_REPO = "sentence-transformers/stackexchange-duplicates"
SE_FILE = "title-title-pair/train-00000-of-00001.parquet"
WILDCHAT_REPO = "allenai/WildChat-1M"


# --------------------------------------------------------------------------
# stackexchange
# --------------------------------------------------------------------------
def build_stackexchange(args, device):
    """Duplicate-question titles -> paraphrase clusters, same recipe as Quora."""
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download

    print("\n=== stackexchange: duplicate titles -> paraphrase clusters ===")
    path = hf_hub_download(SE_REPO, SE_FILE, repo_type="dataset")
    table = pq.read_table(path).to_pydict()
    t1, t2 = table["title1"], table["title2"]
    print(f"  {len(t1)} duplicate title pairs")

    # every row of this corpus is a positive pair -- unlike Quora there is no
    # label column, the dataset is duplicates only
    uf = UnionFind()
    for a, b in zip(t1, t2):
        a, b = a.strip(), b.strip()
        if a and b and a != b:
            uf.union(a, b)

    groups = [g for g in uf.groups().values() if len(g) >= args.min_cluster]
    groups.sort(key=len, reverse=True)
    if args.max_clusters and len(groups) > args.max_clusters:
        rng = np.random.default_rng(0)
        keep = rng.choice(len(groups), size=args.max_clusters, replace=False)
        groups = [groups[i] for i in sorted(keep)]

    texts, cids = [], []
    for cid, members in enumerate(groups):
        for q in sorted(members):
            texts.append(q)
            cids.append(cid)
    cids = np.asarray(cids, dtype="int32")
    sizes = np.array([len(g) for g in groups])
    print(
        f"  kept {len(groups)} clusters (size >= {args.min_cluster}), "
        f"{len(texts)} titles; cluster size mean {sizes.mean():.2f} "
        f"median {np.median(sizes):.0f} max {sizes.max()}"
    )

    print(f"  embedding {len(texts)} titles on {device} ...")
    emb = embed(texts, device, args.batch_size)

    meta = {
        "name": "stackexchange",
        "model": MODEL_NAME,
        "dims": DIM,
        "count": int(emb.shape[0]),
        "n_clusters": int(len(groups)),
        "min_cluster_size": args.min_cluster,
        "has_ground_truth_clusters": True,
        "arrival_order": "synthesised by traces.py (parameterised Zipf/drift)",
        "inputs": [{"repo": SE_REPO, "file": SE_FILE,
                    "sha256": sha256_file(path)}],
    }
    write_trace("stackexchange", emb, cids, meta)
    return emb, cids


# --------------------------------------------------------------------------
# wildchat-long
# --------------------------------------------------------------------------
def build_wildchat_long(args, device):
    """All 14 WildChat shards, so the trace spans the full release window."""
    import pyarrow.parquet as pq
    from huggingface_hub import HfApi, hf_hub_download

    print("\n=== wildchat-long: WildChat-1M, all shards, real arrival order ===")
    api = HfApi()
    info = api.repo_info(WILDCHAT_REPO, repo_type="dataset")
    files = sorted(s.rfilename for s in info.siblings
                   if s.rfilename.endswith(".parquet"))
    print(f"  {len(files)} shards")

    prompts, stamps, inputs = [], [], []
    for fname in files:
        path = hf_hub_download(WILDCHAT_REPO, fname, repo_type="dataset")
        inputs.append({"repo": WILDCHAT_REPO, "file": fname,
                       "sha256": sha256_file(path)})
        pf = pq.ParquetFile(path)
        for batch in pf.iter_batches(
            batch_size=4096, columns=["conversation", "timestamp", "language"]
        ):
            for row in batch.to_pylist():
                if row["language"] != "English":
                    continue
                convo = row["conversation"]
                if not convo or convo[0].get("role") != "user":
                    continue
                text = (convo[0].get("content") or "").strip()
                if not (args.min_chars <= len(text) <= args.max_chars):
                    continue
                prompts.append(text)
                stamps.append(row["timestamp"])
        print(f"  {fname}: running total {len(prompts)} prompts")

    stamps_np = np.asarray(stamps, dtype="datetime64[us]")
    order = np.argsort(stamps_np, kind="stable")

    # Subsample *uniformly along the time axis* rather than truncating, so the
    # trace keeps the full span. Truncating to the first N would just rebuild
    # the two-month `wildchat` corpus with extra download.
    if args.n_wildchat and len(order) > args.n_wildchat:
        sel = np.linspace(0, len(order) - 1, args.n_wildchat).astype(np.int64)
        order = order[sel]
        print(f"  subsampled uniformly in time to {len(order)} prompts")

    prompts = [prompts[i] for i in order]
    span = (str(stamps_np[order[0]]), str(stamps_np[order[-1]]))
    print(f"  {len(prompts)} prompts, span {span[0]} -> {span[1]}")

    print(f"  embedding {len(prompts)} prompts on {device} ...")
    emb = embed(prompts, device, args.batch_size)
    cids = np.full(len(prompts), -1, dtype="int32")

    meta = {
        "name": "wildchat-long",
        "model": MODEL_NAME,
        "dims": DIM,
        "count": int(emb.shape[0]),
        "n_clusters": 0,
        "has_ground_truth_clusters": False,
        "arrival_order": "real, sorted by conversation timestamp (UTC)",
        "time_span": list(span),
        "subsampling": ("uniform along the time axis to preserve span"
                        if args.n_wildchat else "none"),
        "filters": {"language": "English", "turn": "first user message",
                    "min_chars": args.min_chars, "max_chars": args.max_chars},
        "inputs": inputs,
    }
    write_trace("wildchat-long", emb, cids, meta)
    return emb, cids


# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--only", choices=["stackexchange", "wildchat-long"])
    ap.add_argument("--device", default="auto")
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--min-cluster", type=int, default=3)
    ap.add_argument("--max-clusters", type=int, default=0,
                    help="0 = keep all")
    ap.add_argument("--n-wildchat", type=int, default=150_000)
    ap.add_argument("--min-chars", type=int, default=8)
    ap.add_argument("--max-chars", type=int, default=2000)
    args = ap.parse_args()

    device = pick_device(args.device)
    print(f"device: {device}\nmodel : {MODEL_NAME} ({DIM}-d)")
    os.makedirs(DATA_DIR, exist_ok=True)

    if args.only in (None, "stackexchange"):
        build_stackexchange(args, device)
    if args.only in (None, "wildchat-long"):
        build_wildchat_long(args, device)

    print("\ndone. corpora in", DATA_DIR)


if __name__ == "__main__":
    main()

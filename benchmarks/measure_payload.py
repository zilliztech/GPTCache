"""Measure the real payload size of a cached entry, for iso-memory comparison.

Hit rate plotted against *entry count* silently assumes every policy pays the
same price per entry. ARC does not: on top of the ``c`` resident entries it
keeps up to ``c`` ghosts, and this implementation's ghosts carry an embedding
(``dim * 4`` bytes) rather than classic ARC's 8-byte page id. So ARC must be
charged more than ``c``.

How much more is not ``2c``. A resident entry stores the embedding *plus* the
question and the cached response; a ghost stores the embedding alone. The
charge is therefore

    ratio = 1 + ghost_bytes / resident_bytes
          = 1 + emb / (emb + question + response)

which collapses to ``2.0`` only when the payload is empty -- which is exactly
what ``simcache.py`` does, since the simulator holds vectors and no text. The
``2.0`` is an artifact of the harness, not a property of ARC.

This script replaces the guess with a measurement. It re-reads the WildChat
shards named in ``wildchat_meta.json``, applies the same filters, and recovers
the assistant response for every prompt in the trace -- the text GPTCache would
have cached. It writes the resulting byte distribution and the implied
``ratio`` to ``results/payload.json``, which ``analyze.py`` consumes to plot
hit rate against bytes instead of entries.

Quora carries no responses (it is a paraphrase corpus, and its arrival order is
synthesised anyway), so its ratio is reported under WildChat's measured
response distribution and flagged as transferred rather than measured.

Usage::

    python measure_payload.py                 # uses benchmarks/data + results
    python measure_payload.py --out other.json
"""

import argparse
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, "data")
RESULTS_DIR = os.path.join(HERE, "results")

# float32 embedding, the same width prepare_data.py wrote
BYTES_PER_FLOAT = 4


def load_meta(trace):
    with open(os.path.join(DATA_DIR, f"{trace}_meta.json"), encoding="utf-8") as f:
        return json.load(f)


def collect_wildchat(meta):
    """Recover (prompt, response) byte lengths for the prompts in the trace.

    Mirrors ``prepare_data.build_wildchat`` exactly -- same shards, same
    language/turn/length filters, same timestamp sort and truncation -- so the
    rows here correspond one-for-one with the trace's embeddings.
    """
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download

    from prepare_data import sha256_file

    filters = meta["filters"]
    min_chars, max_chars = filters["min_chars"], filters["max_chars"]
    n_target = meta["count"]

    q_bytes, r_bytes, stamps = [], [], []
    n_no_reply = 0

    for entry in meta["inputs"]:
        path = hf_hub_download(entry["repo"], entry["file"], repo_type="dataset")
        # the trace pinned these bytes; if they moved, the payload measured
        # here would not describe the entries the sweep actually ran on
        digest = sha256_file(path)
        if digest != entry["sha256"]:
            raise SystemExit(
                f"{entry['file']} hashes to {digest}, but the trace was built "
                f"from {entry['sha256']}. Rebuild the trace or pin the shard."
            )
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
                if not (min_chars <= len(text) <= max_chars):
                    continue

                # the assistant turn is what the cache would actually store
                reply = ""
                if len(convo) > 1 and convo[1].get("role") == "assistant":
                    reply = (convo[1].get("content") or "").strip()
                if not reply:
                    n_no_reply += 1

                q_bytes.append(len(text.encode("utf-8")))
                r_bytes.append(len(reply.encode("utf-8")))
                stamps.append(row["timestamp"])
        print(f"  {entry['file']}: running total {len(q_bytes)} prompts")

    order = np.argsort(np.asarray(stamps, dtype="datetime64[us]"), kind="stable")
    order = order[:n_target]
    q = np.asarray(q_bytes, dtype="int64")[order]
    r = np.asarray(r_bytes, dtype="int64")[order]
    return q, r, n_no_reply


def describe(arr):
    return {
        "mean": float(arr.mean()),
        "median": float(np.median(arr)),
        "p05": float(np.percentile(arr, 5)),
        "p95": float(np.percentile(arr, 95)),
        "max": float(arr.max()),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", default=os.path.join(RESULTS_DIR, "payload.json"))
    args = ap.parse_args()

    meta = load_meta("wildchat")
    dim = meta["dims"]
    emb_bytes = dim * BYTES_PER_FLOAT

    print("=== wildchat: recovering cached-response sizes ===")
    q, r, n_no_reply = collect_wildchat(meta)
    if len(q) != meta["count"]:
        raise SystemExit(
            f"recovered {len(q)} prompts but the trace holds {meta['count']}; "
            "the filters in measure_payload.py have drifted from prepare_data.py"
        )
    print(f"  matched {len(q)} prompts ({n_no_reply} with no assistant reply)")

    # total memory of a c-entry cache is c * mean(bytes), so the mean -- not
    # the median -- is the statistic that sets the charge. Responses are
    # right-skewed, so the two differ noticeably; both are reported.
    resident_mean = emb_bytes + float(q.mean()) + float(r.mean())
    ratio = 1.0 + emb_bytes / resident_mean

    payload = {
        "embedding_bytes": emb_bytes,
        "embedding_dim": dim,
        "n_entries": int(len(q)),
        "n_without_reply": int(n_no_reply),
        "question_bytes": describe(q),
        "response_bytes": describe(r),
        "resident_bytes_mean": resident_mean,
        "ghost_bytes": emb_bytes,
        "ratio": ratio,
        "ratio_note": (
            "ARC(c) holds c resident entries plus up to c embedding-only "
            "ghosts; charge ARC ratio*c entries when comparing against a "
            "policy that keeps no ghosts."
        ),
        "traces": {
            "wildchat": {"ratio": ratio, "source": "measured"},
            "quora-stationary": {"ratio": ratio, "source": "transferred"},
            "quora-drift": {"ratio": ratio, "source": "transferred"},
        },
        "transferred_note": (
            "Quora is a paraphrase corpus with no assistant responses, so its "
            "entries are charged under WildChat's measured response "
            "distribution rather than one of its own."
        ),
    }

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
        f.write("\n")

    print()
    print(f"  embedding      {emb_bytes} B")
    print(f"  question mean  {q.mean():10.1f} B   median {np.median(q):8.1f} B")
    print(f"  response mean  {r.mean():10.1f} B   median {np.median(r):8.1f} B")
    print(f"  resident mean  {resident_mean:10.1f} B")
    print(f"  ghost          {emb_bytes} B")
    print(f"  ARC charge     {ratio:.3f}c   (2.000c would assume an empty payload)")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()

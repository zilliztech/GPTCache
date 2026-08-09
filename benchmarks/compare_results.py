"""Compare regenerated results against a reference run held in git.

benchmarks/results/ is not committed on this branch, so there is no reference at
HEAD until you commit one: pass ``--ref`` naming a revision that has the results.

``git status benchmarks/results/`` is *not* a reproducibility check. Three of the
artifacts move on every machine no matter how exact the run is:

* ``summary.csv`` and ``summary.md`` carry latency, throughput and heap columns,
  which are wall-clock measurements of the host;
* the PNGs differ in encoding bytes across matplotlib builds even when every
  plotted value is identical.

What must match exactly is the correctness metrics -- hit rate, false hit rate,
the paired-bootstrap intervals derived from them, and the cache occupancy
counters. This script separates the two and reports them apart.

    python benchmarks/compare_results.py --ref <commit-holding-results>
    python benchmarks/compare_results.py            # once results/ is committed
    python benchmarks/compare_results.py --file benchmarks/results/summary.csv

Needs a git checkout: the reference is read with ``git show``, so this runs on
the host, not inside the benchmark container (the build context omits .git).

Exit status is 0 when every exact column matches, 1 otherwise. Latency
differences never affect the exit status; they are printed for information.
"""

import argparse
import io
import os
import subprocess
import sys

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
DEFAULT_FILE = "benchmarks/results/summary.csv"

# Columns that must be bit-for-bit identical. Everything the paper claims is
# built out of these.
EXACT = [
    "hit_rate", "delta_vs_lru_pp", "ci_lo_pp", "ci_hi_pp", "significant",
    "false_hit_rate", "n_hits", "n_false_hits", "policy_vectors",
    "resident_mean", "p_final", "p_max", "p_mean",
]

# Wall-clock and memory measurements of the host. Expected to move.
HARDWARE = [
    "lat_mean_us", "lat_p50_us", "lat_p95_us", "lat_p99_us",
    "evict_us_per_req", "throughput_qps", "wall_seconds", "peak_heap_kb",
]

KEY_CANDIDATES = ["trace", "policy", "capacity", "seed", "corpus",
                  "experiment", "drift_rate", "zipf_s", "stride"]


def committed(path, ref):
    out = subprocess.run(["git", "show", f"{ref}:{path}"], cwd=REPO,
                         capture_output=True, text=True)
    if out.returncode != 0:
        sys.exit(
            f"cannot read {path} at {ref}: {out.stderr.strip()}\n"
            f"benchmarks/results/ is not committed on this branch, so {ref} has "
            f"no reference to compare against.\n"
            f"Pass --ref <commit> naming a revision that holds the results, or "
            f"commit this run's results/ and re-run without --ref."
        )
    return pd.read_csv(io.StringIO(out.stdout))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--file", default=DEFAULT_FILE,
                    help=f"repo-relative CSV to compare (default {DEFAULT_FILE})")
    ap.add_argument("--ref", default="HEAD",
                    help="git revision holding the reference (default HEAD)")
    args = ap.parse_args()

    live_path = os.path.join(REPO, args.file)
    if not os.path.exists(live_path):
        sys.exit(f"{args.file} not found -- run the pipeline first")

    old = committed(args.file, args.ref)
    new = pd.read_csv(live_path)
    keys = [c for c in KEY_CANDIDATES if c in old.columns and c in new.columns]
    if not keys:
        sys.exit(f"no key columns found in {args.file}")

    print(f"reference : {args.ref}:{args.file}  ({len(old)} rows)")
    print(f"regenerated: {args.file}  ({len(new)} rows)")
    print(f"keyed on  : {', '.join(keys)}\n")

    merged = old.merge(new, on=keys, suffixes=("_old", "_new"), how="outer",
                       indicator=True)
    unmatched = merged[merged["_merge"] != "both"]
    both = merged[merged["_merge"] == "both"]

    def present(names):
        return [c for c in names if f"{c}_old" in merged.columns]

    failures = []
    print("must match exactly")
    print("-" * 64)
    for col in present(EXACT):
        a, b = both[f"{col}_old"], both[f"{col}_new"]
        if pd.api.types.is_bool_dtype(a) or a.dtype == object:
            n = int((a != b).sum())
            print(f"  {col:<20} {'mismatched rows':>18}: {n}")
            if n:
                failures.append(col)
            continue
        d = (pd.to_numeric(a, errors="coerce")
             - pd.to_numeric(b, errors="coerce")).abs().max()
        d = 0.0 if pd.isna(d) else float(d)
        print(f"  {col:<20} {'max |delta|':>18}: {d:.10g}")
        if d != 0.0:
            failures.append(col)

    hw = present(HARDWARE)
    if hw:
        print("\nhardware-dependent, expected to differ (not part of the verdict)")
        print("-" * 64)
        for col in hw:
            d = (pd.to_numeric(both[f"{col}_old"], errors="coerce")
                 - pd.to_numeric(both[f"{col}_new"], errors="coerce")).abs().max()
            print(f"  {col:<20} {'max |delta|':>18}: "
                  f"{0.0 if pd.isna(d) else d:.6g}")

    print()
    if len(unmatched):
        print(f"MISMATCH: {len(unmatched)} rows exist on only one side "
              f"(grid changed)")
        print(unmatched[keys + ["_merge"]].head(10).to_string(index=False))
        return 1
    if failures:
        print(f"MISMATCH: {len(failures)} correctness column(s) moved: "
              f"{', '.join(failures)}")
        return 1
    print(f"MATCH: every correctness column identical across {len(both)} rows.")
    print("Latency, throughput and heap differ because they measure this host.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

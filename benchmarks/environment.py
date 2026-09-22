"""Record the environment a benchmark run happened in.

Writes ``benchmarks/results/environment.json``. This is the artefact that makes a
result auditable after the fact: it pins down the three things that can move a
number without anyone touching the code -- the source revision, the CPU
architecture and BLAS thread count, and the exact package versions -- plus the
sha256 of every input parquet, lifted from the ``data/*_meta.json`` that
``prepare_data.py`` already writes.

Run standalone, or at the end of the pipeline:

    python benchmarks/environment.py
    python benchmarks/environment.py --out somewhere/else.json

The git revision comes from ``git`` when a checkout is visible, or from the
``GIT_COMMIT`` / ``GIT_DIRTY`` environment variables when they are set. Inside
the container neither applies -- ``.dockerignore`` excludes ``.git`` -- so the
revision is recorded as null. Run this script on the host for a stamped one.
"""

import argparse
import glob
import json
import os
import platform
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
# Overridable via BENCH_DATA_DIR for running this script outside the container
# against corpora kept elsewhere. The pipeline never sets it: one corpus size
# means one set of reference numbers.
DATA_DIR = (os.environ.get("BENCH_DATA_DIR")
            or os.path.join(HERE, "data"))
# Overridable so a reduced run (smoke/quick) cannot overwrite the committed
# reference artifacts; run_pipeline.sh points those at a scratch subdirectory.
RESULTS_DIR = (os.environ.get("BENCH_RESULTS_DIR")
               or os.path.join(HERE, "results"))

# Everything whose version can move a measured number. cachetools is in the list
# because it backs LRU/LFU/FIFO/RR; huggingface-hub because it fetches the data.
PACKAGES = [
    "torch",
    "numpy",
    "cachetools",
    "transformers",
    "sentence-transformers",
    "huggingface-hub",
    "datasets",
    "pyarrow",
    "pandas",
    "matplotlib",
    "faiss-cpu",
    "SQLAlchemy",
    "pytest",
]

# Thread-count variables change BLAS reduction order, so they change embeddings.
THREAD_VARS = [
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "PYTHONHASHSEED",
]


def _git(*args):
    try:
        out = subprocess.run(
            ["git", *args], cwd=HERE, capture_output=True, text=True, timeout=10
        )
        return out.stdout.strip() if out.returncode == 0 else None
    except (OSError, subprocess.SubprocessError):
        return None


def git_revision():
    """``(commit, dirty)``. Env wins, because the image has no ``.git``."""
    commit = os.environ.get("GIT_COMMIT") or _git("rev-parse", "HEAD")
    dirty_env = os.environ.get("GIT_DIRTY")
    if dirty_env is not None:
        dirty = dirty_env.strip().lower() in ("1", "true", "yes", "dirty")
    else:
        status = _git("status", "--porcelain")
        dirty = bool(status) if status is not None else None
    return {"commit": commit, "dirty": dirty}


def package_versions():
    import importlib.metadata as md

    out = {}
    for name in PACKAGES:
        try:
            out[name] = md.version(name)
        except md.PackageNotFoundError:
            out[name] = None
    return out


def corpora():
    """Per-corpus provenance from ``data/*_meta.json``: hashes, counts, embed args."""
    out = {}
    for path in sorted(glob.glob(os.path.join(DATA_DIR, "*_meta.json"))):
        name = os.path.basename(path)[: -len("_meta.json")]
        try:
            with open(path) as fh:
                meta = json.load(fh)
        except (OSError, ValueError) as exc:
            out[name] = {"error": str(exc)}
            continue
        keep = ("count", "model", "dims", "n_clusters", "min_cluster_size",
                "arrival_order", "span", "inputs", "embed")
        out[name] = {k: meta[k] for k in keep if k in meta}
    return out


def tau():
    path = os.path.join(DATA_DIR, "tau.json")
    if not os.path.exists(path):
        return None
    try:
        with open(path) as fh:
            data = json.load(fh)
    except (OSError, ValueError):
        return None
    return data.get("tau", data) if isinstance(data, dict) else data


def blas_info():
    """Which BLAS numpy is bound to, and which SIMD level it selected.

    Both change floating-point reduction order, so both change embeddings at the
    1e-7 level. This is the concrete reason a macOS run (Accelerate) and a
    linux/amd64 container run (OpenBLAS) are not expected to be bit-identical.
    """
    try:
        import numpy as np

        cfg = np.show_config(mode="dicts")
        blas = cfg.get("Build Dependencies", {}).get("blas", {})
        return {
            "name": blas.get("name"),
            "version": blas.get("version"),
            "detection method": blas.get("detection method"),
            "simd_extensions": cfg.get("SIMD Extensions"),
        }
    except Exception:  # pragma: no cover - purely informational
        return None


def build(mode=None):
    return {
        "git": git_revision(),
        "mode": mode or os.environ.get("BENCH_MODE"),
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "python": platform.python_version(),
            "in_container": os.path.exists("/.dockerenv"),
            "cpu_count": os.cpu_count(),
        },
        "threading": {v: os.environ.get(v) for v in THREAD_VARS},
        "blas": blas_info(),
        "packages": package_versions(),
        "tau": tau(),
        "corpora": corpora(),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", default=os.path.join(RESULTS_DIR, "environment.json"))
    ap.add_argument("--mode", default=None, help="pipeline mode that produced the run")
    args = ap.parse_args()

    manifest = build(args.mode)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(manifest, fh, indent=2, sort_keys=True)
        fh.write("\n")

    git = manifest["git"]
    print(f"wrote {args.out}")
    print(f"  commit   : {git['commit']} (dirty={git['dirty']})")
    print(f"  platform : {manifest['platform']['system']}/"
          f"{manifest['platform']['machine']} python "
          f"{manifest['platform']['python']}")
    print(f"  threads  : OMP={manifest['threading'].get('OMP_NUM_THREADS')}")
    print(f"  corpora  : {', '.join(manifest['corpora']) or 'none'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

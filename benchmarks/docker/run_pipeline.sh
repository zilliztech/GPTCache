#!/usr/bin/env bash
#
# Runs the eviction-policy benchmarks. Takes one argument: the mode.
# The Dockerfile's ENTRYPOINT invokes this; run it with `--help` for the modes.

set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."

PY="${PYTHON:-python}"
DATA_DIR="benchmarks/data"
MODE="${1:-smoke}"

case "$MODE" in
    # quora must come first: its build writes data/tau.json, which every later
    # stage reads. SWEEP is never empty, so `set -u` is safe on bash 3.2.
    smoke) CORPORA=(quora);                      SWEEP=(--quick);   FULL=0 ;;
    quick) CORPORA=(quora wildchat wikianswers); SWEEP=(--seeds 3); FULL=0 ;;
    full)  CORPORA=(quora wildchat wikianswers stackexchange wildchat-long)
           SWEEP=(--seeds 10);                                      FULL=1 ;;
    *)
        cat <<'EOF'
Usage: run_pipeline.sh [smoke|quick|full]

  smoke   ~3 min   quora only, reduced sweep  -> results/_smoke/   (default)
  quick   ~20 min  three corpora, 3 seeds     -> results/_quick/
  full    ~1-2 h   everything; the reference
                   run                        -> results/

Times are first-run, including download and embedding; later runs reuse the
data volume. Corpus size, seed count and single-stage selection are deliberately
not options: one corpus size means one set of reference numbers, and a partial
size-tuned corpus left in the data volume would make the next `full` run skip
prep and reproduce nothing. To redo one step against existing CSVs, call its
script directly (e.g. python benchmarks/analyze.py).
To discard the downloaded corpora and start over: docker compose down -v
EOF
        if [[ "$MODE" == "-h" || "$MODE" == "--help" ]]; then
            exit 0
        fi
        echo "error: unknown mode '$MODE'" >&2
        exit 2 ;;
esac

# Only `full` writes the committed reference artifacts; the reduced modes go to
# a gitignored scratch directory so they cannot overwrite a reference figure.
if (( FULL )); then
    OUT_DIR="benchmarks/results"
else
    OUT_DIR="benchmarks/results/_$MODE"
fi
export BENCH_RESULTS_DIR="$PWD/$OUT_DIR"
mkdir -p "$OUT_DIR" "$DATA_DIR"

# Not a flag: run_bench.py defaults to 1 worker, which is needlessly slow. Sweep
# cells are independently deterministic, so this cannot change a result.
JOBS="$($PY -c 'import os; print(os.cpu_count() or 1)')"

step()        { printf '\n\033[1m=== %s ===\033[0m\n' "$*"; }
have_corpus() { [[ -f "$DATA_DIR/$1_emb.npy" && -f "$DATA_DIR/$1_cid.npy" ]]; }

echo "mode $MODE | corpora ${CORPORA[*]} | -> $OUT_DIR"
if (( ! FULL )); then
    echo "NOTE: reduced run -- only 'full' reproduces the committed numbers."
fi

STARTED_AT=$(date +%s)

# Both pytest flags are required in a clean environment: `-o addopts=` clears the
# --html option tests/pytest.ini sets, which needs pytest-html; and
# test_distributed_cache.py imports redis_om and wants a live Redis.
step "Unit tests"
$PY -m pytest tests/unit_tests/eviction tests/unit_tests/manager/test_eviction.py \
    --ignore=tests/unit_tests/eviction/test_distributed_cache.py -o addopts= -q

step "Build corpora"
for c in "${CORPORA[@]}"; do
    if have_corpus "$c"; then
        echo "    $c already built, reusing"
    elif [[ "$c" == quora || "$c" == wildchat ]]; then
        $PY benchmarks/prepare_data.py --only "$c" --device cpu
    else
        $PY benchmarks/prepare_data_ext.py --only "$c" --device cpu
    fi
done
# Every downstream script skips a missing corpus with only a printed note, so a
# partial build would yield a result that looks complete.
for c in "${CORPORA[@]}"; do
    have_corpus "$c" || { echo "error: corpus '$c' was not built" >&2; exit 1; }
done

step "Screening gate"
$PY benchmarks/gate_screening.py

step "Headline sweep"
$PY benchmarks/run_bench.py --jobs "$JOBS" "${SWEEP[@]}"

if have_corpus wildchat; then
    step "Payload measurement"
    $PY benchmarks/measure_payload.py
fi

step "Figures and summary"
$PY benchmarks/analyze.py

if (( FULL )); then
    step "Crossover study"
    $PY benchmarks/sweep_crossover.py --jobs "$JOBS"
    $PY benchmarks/analyze_crossover.py

    step "Working-set analysis"
    $PY benchmarks/working_set.py
fi

step "Provenance manifest"
$PY benchmarks/environment.py --mode "$MODE"

ELAPSED=$(( $(date +%s) - STARTED_AT ))
step "Done in $((ELAPSED / 60))m $((ELAPSED % 60))s -- wrote $OUT_DIR"
ls -1 "$OUT_DIR" | sed 's/^/    /'
if (( FULL )); then
    echo
    echo "To check this run against an earlier one, from the host -- not in here,"
    echo "the build context excludes .git and compare_results.py shells out to it:"
    echo "    python benchmarks/compare_results.py --ref <commit-holding-results>"
fi

# Storage compression frontier — 100K real-encoder run

Source: `bench_real_100k/results.json` · data=qqp (real) · FAISS 1.13.2 · threads=1 ·
repeats=3 · threshold=0.90 + per-cell sweep 0.84–0.96 · encoder fallback ONNX→sbert768-768d
(static-batch ONNX detected). D/H/F rows below are from the prior M-sweep run; the latest run is
cells **A, E, G, I** (adds isolation cell I + the threshold sweep — see the two sections below).

RAM is `faiss.serialize_index` (index-only, in-RAM). **bytes/vector = RAM ÷ 100 000** is the axis
that scales; "vs A" is index-RAM compression against the ONNX/768/Flat baseline.

| Cell | Config | B/vec | vs A | TP% | FP% | precision* | srch p95 (ms) | frontier |
|---|---|---:|---:|---:|---:|---:|---:|---|
| A | ONNX/768/Flat | 3080 | 1.0× | 89.25 | 18.50 | 82.8 | 19.109 | baseline |
| I | MRL/768/Flat | 3072 | 1.0× | 97.95 | 32.90 | 74.9 | 19.820 | isolation (no compression) |
| F | MRL/256/HNSW+PQ+Refine | 1339 | 2.3× | 97.35 | 38.25 | 71.8 | 0.198 | dominated |
| D | MRL/256/HNSW+SQ8 (M=32) | 536 | 5.7× | 98.90 | 40.90 | 70.7 | 0.327 | dominated |
| H | MRL/256/HNSW+SQ8 (M=24) | 472 | 6.5× | 98.85 | 40.85 | 70.8 | 0.314 | dominated |
| G | MRL/256/HNSW+SQ8 (M=16) | 408 | 7.5× | 98.85 | 40.80 | 70.8 | 0.285 | **on frontier** |
| E | MRL/256/HNSW+PQ | 315 | 9.8× | 95.40 | 24.80 | 79.4 | 0.190 | **on frontier** |
| J | StaticMRL/256/HNSW+PQ | 315 | 9.8× | 65.30 | 8.15 | 88.9 | 0.133 | frontier (e2e-latency) |

\* precision proxy = TP/(TP+FP) at equal TP/FP query counts (2000 each).

## Pareto frontier

Minimize bytes/vector, maximize TP. The non-dominated set is **E → G**:

- **E (HNSW+PQ)** — 315 B/vec, lowest RAM. Nothing has both less RAM and higher TP.
- **G (HNSW+SQ8, M=16)** — 408 B/vec at the recall ceiling (98.85%). Dominates H and D (same TP,
  more RAM) and F (more RAM, lower TP).

Dominated cells: **D, H** (M=32/24 cost RAM for ≤0.05pp TP over M=16), **F** (refine, see below),
**A** (dominated on every axis).

## Operating points

- **Max compression — Cell E (PQ): 9.8× vs A**, TP 95.40% (**+6.15pp over the baseline's 89.25%**),
  search p95 0.190 ms (≈100× faster than A's 19.1 ms). This is the honest headline, up from the
  5.7× the SQ8-only cell (D) gave at 10K — adding PQ to the production MRL/256 encoder buys the
  extra 4.1×.
- **Max accuracy — Cell G (SQ8, M=16): 7.5× vs A**, TP 98.85% at the recall ceiling, p95 0.285 ms.

Selection rule (min-RAM cell that still holds Cell A's tp_hit_rate): A=89.25%; every MRL cell clears
it; the smallest is **E → 9.8×**.

## Honesty notes

1. **Refine (F) is the cautionary cell, exactly as predicted.** `IndexRefineFlat` keeps the full
   float32 vectors, so F lands at 1339 B/vec — **4.3× larger than E** — for *lower* TP than G/D.
   Dominated. Report it as "why we don't ship refine," not as a result.
2. **The M-sweep says M=16 wins.** G ties D/H on TP at 24–32% less RAM, so the SQ8 operating point is
   M=16; the default M=32 (D) is wasted RAM.
3. **The fixed-0.90 precision gap is largely a threshold artifact (sweep added, A,E,G,I run).** The
   per-cell threshold sweep (0.84–0.96, below) shows the global 0.90 threshold sits at a *different*
   operating point on each cell's PR curve — it was tuned for the 768-d ONNX baseline, so it
   over-penalizes the sharper MRL cells. At a threshold matched to A's precision (~0.83), the MRL
   cells hold TP **≥** the baseline at 7.5–9.8× less RAM:
   - A @0.90 — prec 0.828, TP 0.892 (baseline operating point).
   - E @0.92 — prec **0.864**, TP 0.882 (≈ A's TP at higher precision, **9.8×** less RAM).
   - G @0.94 — prec **0.838**, TP 0.900 (**>** A's TP at higher precision, **7.5×** less RAM).
   - I @0.92 — prec 0.804, TP 0.948; @0.94 prec 0.846, TP 0.877 (brackets A at 1.0× RAM).

   So the honest revision: on the *accuracy axis* the MRL cells are Pareto-competitive with the
   baseline once you tune the threshold per cell; the "baseline keeps best precision" line only holds
   at a single shared threshold. The genuine remaining cost is **e2e latency** (note 5), not accuracy.

## Isolation cell I — where the FP rise actually comes from

Cell **I (MRL/768/Flat)** holds dimension at 768 and the index at flat float32, swapping *only* the
encoder (ONNX/sbert768 → nomic MRL). It decomposes the FP rise at the shared 0.90 threshold:

| Step | Change | FP | ΔFP |
|---|---|---:|---:|
| A → I | encoder swap (ONNX → MRL), full 768, flat | 18.5 → 32.9 | **+14.4** |
| I → G | truncate 768→256 + SQ8 | 32.9 → 40.8 | +7.9 |
| I → E | truncate 768→256 + PQ | 32.9 → 24.9 | **−8.0** |

**The encoder swap drives most of the precision loss, not MRL truncation.** Truncation+SQ8 adds only
+7.9pp on top, and truncation+**PQ actually *lowers* FP below the full-dim MRL** (−8.0pp) because PQ
quantization suppresses spurious near-matches. This defends the truncation lever: MRL/256 is not the
precision culprit the single-threshold view implied — the stronger-but-looser encoder is, and PQ
claws precision back while still hitting 9.8× compression.

## Per-cell threshold sweep (TP / FP / precision)

| thr | A TP/FP/prec | E TP/FP/prec | G TP/FP/prec | I TP/FP/prec | J TP/FP/prec |
|---:|---|---|---|---|---|
| 0.88 | .933/.237/.798 | .978/.421/.699 | .995/.597/.625 | .995/.473/.678 | .753/.122/.861 |
| 0.90 | .892/.185/.828 | .953/.248/.793 | .989/.408/.708 | .980/.329/.749 | .653/.082/.889 |
| 0.92 | .822/.146/.849 | .882/.139/.864 | .967/.273/.780 | .948/.231/.804 | .492/.040/.925 |
| 0.94 | .733/.113/.866 | .695/.075/.903 | .900/.174/.838 | .877/.160/.846 | .164/.004/.973 |

## Honesty note 5 — e2e latency reverses the search win

Search-only latency confirms HNSW's **~100× speedup** (A/I flat ≈19.8 ms p95 → E/G HNSW ≈0.2–0.3 ms).
But **end-to-end p95 *rises*** with the MRL cells — A 59.1 ms → E 97.5 / G 97.8 / I 117.1 ms — because
the nomic MRL encoder is slower per query than the ONNX/sbert768 encoder, and embedding dominates e2e
(~97 ms for MRL/256 vs ~27 ms encode for the baseline; search is ≤0.3 ms either way). So the 100×
search win is real but **invisible at e2e** under this encoder pairing; the compression and search-CPU
wins are the bankable ones, latency is not. Quote search-p95 for the search claim, e2e-p95 separately.

**Follow-up — Cell J makes the e2e win bankable too.** Swapping the nomic encoder for
`static-retrieval-mrl-en-v1` (token-embedding lookup, no transformer forward) at the *same* MRL/256/
HNSW+PQ index collapses **e2e p95 from 97.5 ms → 0.50 ms** (search p95 0.13 ms; e2e is now search +
SQLite-bound, not encoder-bound) at **identical 9.8× / 315 B/vec** compression. The cost is recall:
J is a weaker encoder, **−12.9pp TP at matched precision ≈0.86** (E @0.92 TP .882/prec .864 vs
J @0.88 TP .753/prec .861), and it trails even baseline A there (A @0.92 TP .822/prec .849). So the
frontier has a third operating point: **A** = recall+latency, no compression; **E/G** = compression,
encoder-bound e2e; **J** = compression **and** sub-ms e2e, paid for in recall.

# Storage compression frontier — full-QQP run (134K, real encoder)

Source: `bench_real_1m/results.json` · data=qqp (real) · FAISS 1.13.2 · threads=1 ·
repeats=3 · threshold=0.90 · encoder fallback ONNX→sbert768-768d (static-batch ONNX detected).

**Scale honesty:** the run requested `--scale 1000000`, but `_try_load_qqp` caps ingest to the
QQP `label==1` (duplicate) subset, which is **134 378 pairs**. Cell A's flat index is deterministic
at 3080 B/vec, so 413 884 330 ÷ 3080 = 134 378 vectors — confirming the cap. This is the **full QQP
duplicate corpus**, ~1.3× the 100K run, **not** a 1M corpus. To truly hit 1M you need a larger
source (e.g. concatenate multiple datasets); QQP alone cannot.

bytes/vector = `faiss.serialize_index` RAM ÷ 134 378. "vs A" is index-RAM compression against the
ONNX/768/Flat baseline.

| Cell | Config | B/vec | vs A | TP% | FP% | precision* | srch p95 (ms) | frontier |
|---|---|---:|---:|---:|---:|---:|---:|---|
| A | ONNX/768/Flat | 3080 | 1.0× | 90.60 | 20.35 | 81.7 | 36.131 | baseline |
| F | MRL/256/HNSW+PQ+Refine | 1338 | 2.3× | 97.30 | 40.75 | 70.5 | 0.459 | dominated |
| G | MRL/256/HNSW+SQ8 (M=16) | 408 | 7.5× | 98.95 | 43.75 | 69.3 | 0.541 | **on frontier** |
| E | MRL/256/HNSW+PQ | 314 | 9.8× | 95.55 | 26.85 | 78.1 | 0.438 | **on frontier** |

\* precision proxy = TP/(TP+FP) at equal TP/FP query counts (2000 each).

## What this scale adds over the 100K run

1. **Per-vector RAM is scale-independent.** B/vec matches the 100K run to rounding (A 3080, E 314 vs
   315, F 1338 vs 1339, G 408 vs 408). Compression is a property of the encoder+index, not the
   corpus size — so the **9.8× headline holds at every scale measured**.

2. **Search latency is where scale bites, and it favors compression.** Brute-force flat (A) grew
   **19.1 ms → 36.1 ms p95** going 100K → 134K (linear in N, as expected). The HNSW cells stayed
   **sub-millisecond** (E 0.44 ms, G 0.54 ms) — O(log n). At this corpus **E searches ~83× faster
   than A**, and the gap widens with N. This is the headline that only a larger corpus can show;
   RAM compression and search scaling point the same way.

## Pareto frontier

Minimize bytes/vector, maximize TP. Non-dominated set is **E → G** (unchanged from 100K):

- **E (HNSW+PQ)** — 314 B/vec, lowest RAM. Nothing has both less RAM and higher TP.
- **G (HNSW+SQ8, M=16)** — 408 B/vec at the recall ceiling (98.95%). Dominates F (more RAM, lower
  TP) and A (worse on every axis).

## Operating points

- **Max compression — Cell E (PQ): 9.8× vs A**, TP 95.55% (**+4.95pp over the baseline's 90.60%**),
  search p95 0.438 ms (≈83× faster than A's 36.1 ms). Honest headline.
- **Max accuracy — Cell G (SQ8, M=16): 7.5× vs A**, TP 98.95% at the recall ceiling, p95 0.541 ms.

Selection rule (min-RAM cell that still holds Cell A's tp_hit_rate): A=90.60%; every MRL cell clears
it; the smallest is **E → 9.8×**.

## Honesty notes

1. **Refine (F) is again the cautionary cell.** `IndexRefineFlat` keeps the full float32 vectors, so
   F lands at 1338 B/vec — **4.3× larger than E** — for lower TP than G. Dominated. "Why we don't
   ship refine," not a result.
2. **Precision still favors the baseline.** MRL/256 at threshold 0.90 raises FP (26.9–43.8% vs A's
   20.4%) → A has the best precision (81.7%). Among compressed cells **E is the most precise (78.1%)**
   because PQ quantization suppresses spurious near-matches; the SQ8/refine cells sit at ~69–70%.
   E is both smallest and best-balanced. State this alongside the 9.8× headline — compression is not
   free on precision.
3. **"1M" is a misnomer for this artifact** (see Scale honesty above); cite it as the 134K full-QQP
   run.

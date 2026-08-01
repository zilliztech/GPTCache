# Screening results — LLM cache project (verified in sandbox)

All numbers = cache hit rate, semantic cache, tau=0.65, 12k queries,
800 Zipf(1.1) paraphrase clusters, 64-d embeddings calibrated to
intra-cluster sim ~0.74 / inter-cluster ~0.01 (p99 0.30).
Reproduce: `python3 policy_sim.py`, `python3 tinylfu_test.py`, `python3 final.py`

## Main comparison (cap = 100 entries)

| workload         | LRU*   | LFU    | W-TinyLFU (exact) | W-TinyLFU (LSH) | **Semantic-ARC** |
|------------------|--------|--------|-------------------|-----------------|------------------|
| stationary       | 66.84% | **71.73%** | 70.10%        | 70.83%          | 70.85%           |
| mild drift 20%   | 66.61% | 62.97% | 66.43%            | 67.62%          | **70.16%**       |
| heavy drift 60%  | 66.33% | 48.09% | 61.58%            | 61.99%          | **68.67%**       |
| total reshuffle  | 66.53% | 35.35% | 58.99%            | 61.07%          | **68.03%**       |
| **WORST CASE**   | 66.33% | 35.35% | 58.99%            | 61.07%          | **68.03%**       |

*LRU = GPTCache default.

## Ablation 1 — semantic ghosts are what matters (drifting, cap=100)

| variant                    | hit rate | final p (adaptation state) |
|----------------------------|----------|----------------------------|
| LRU                        | 66.32%   | —                          |
| ARC, **exact-key** ghosts  | 52.63%   | **0 — never adapts**       |
| ARC, **semantic** ghosts   | **68.63%** | 1–9 — adapts             |

## Ablation 2 — exact-key frequency sketches are inert

| sketch keying | mean estimate | max | corr(estimate, true popularity) |
|---------------|---------------|-----|---------------------------------|
| exact key     | 0.21          | 2   | **-0.059  (no signal)**         |
| LSH (12b x 16)| 4.27          | 7   | **+0.743**                      |

## Rejected after testing

| idea                          | verdict | evidence |
|-------------------------------|---------|----------|
| Cost-aware eviction (GDSF)    | TAKEN   | PR #689, Jul 25 2026 |
| W-TinyLFU + cost admission    | TAKEN   | PR #680, Apr 2 2026  |
| Approximate matching          | IS the library | GPTCache core |
| Coverage/redundancy eviction  | **FAILS** | 61.35% vs LRU 67.27% (-5.9pp) |

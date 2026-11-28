# D1 (N1): capacity versus policy — what policy + allocation is worth in slots and MiB

**Conclusion: policy and capacity are not alternatives — at every budget the policy is a free capacity increase, and combined
with allocation it is worth 8–18 % more LRU slots at B/K 1.5–2.** At B=96 the EMA alone equals LRU with 5–6 % more slots, the
normalised power law (C1) 6–7 %, the global percentile pool (C4, no offline curves) 8–10 % and offline allocation + EMA 8–10 %; at
B=128: 6–7 %, 8–9 %, 12–13 % and 16–18 %. Per sequence at 32K context this is 10–50 MiB of GPU cache that need not be held for the
same miss rate (e.g. 8B, B=128: offline allocation + EMA 50 MiB, pool 36 MiB). At fixed memory the same policies cut misses — and
host→GPU bytes — by 10–24 %. N1 ("Bélády is worth 18–46 % more slots, EMA only 5–7 %") still holds for the EMA alone; the gap to
Bélády narrows by a factor of ~2 once allocation is added, and at B=256 the global pool is worth 20–25 % more slots.

Source: `analysis_v3/d1_iso.py` (iso-miss capacity: the uniform LRU budget, interpolated on a 64–1024 step-8 grid, with the same
total misses; `results_v3/d1/{dev,8b}/iso.csv`; LRU fine grids `results_v3/d1/lru_fine_*/misses.csv.gz` from `analysis_v3.run
lru_fine`), policy misses from B1 (EMA with the pooled-best α, C1 with γ=1) and C4 (offline LODO allocation + EMA; LODO-tuned
pool). Memory: one 16-token page of one layer = 64 KiB (K and V, 8 KV heads × 128 dims, bf16) for both models; MiB saved per
sequence = (iso-miss budget − B) × layers × 64 KiB. 14 dev documents per model.

| Model, B (B/K) | EMA | C1 power law | C4 global pool | offline allocation + EMA |
|---|---|---|---|---|
| 0.6B, 80 (1.25) | 1.043 · 6 MiB · −8.4 % | 1.042 · 6 · −8.3 % | 1.045 · 6 · −8.9 % | 1.048 · 7 · −9.3 % |
| 0.6B, **96 (1.5)** | **1.058 · 10 MiB · −10.0 %** | **1.067 · 11 · −11.5 %** | **1.082 · 14 · −14.2 %** | **1.084 · 14 · −14.5 %** |
| 0.6B, 128 (2) | 1.070 · 16 · −10.4 % | 1.090 · 20 · −13.0 % | 1.130 · 29 · −18.0 % | 1.155 · 35 · −20.6 % |
| 0.6B, 256 (4) | 1.068 · 31 · −6.8 % | 1.079 · 35 · −7.8 % | 1.197 · 88 · −16.7 % | 1.055 · 25 · −5.6 % (grid-limited) |
| 8B, 80 (1.25) | 1.039 · 7 · −8.0 % | 1.037 · 7 · −7.5 % | 1.052 · 9 · −10.7 % | 1.052 · 9 · −10.8 % |
| 8B, **96 (1.5)** | **1.052 · 11 · −9.5 %** | **1.061 · 13 · −11.2 %** | **1.096 · 21 · −17.1 %** | **1.101 · 22 · −17.6 %** |
| 8B, 128 (2) | 1.061 · 18 · −9.9 % | 1.078 · 23 · −12.3 % | 1.124 · 36 · −18.5 % | 1.175 · 50 · −24.3 % |
| 8B, 256 (4) | 1.063 · 36 · −6.2 % | 1.064 · 37 · −6.3 % | 1.245 · 141 · −18.6 % | 1.042 · 24 · −4.3 % (grid-limited) |

Cells: iso-miss capacity ratio · MiB of GPU cache saved per sequence at the same misses · miss (= byte) change against LRU at the
same memory. "grid-limited": the offline allocation's grid stops at 256 slots per layer, so at B=256 it is uniform (and uses the
α grid of C4, 0.3–0.7, hence slightly below the EMA row).

Reading:
- **Gain against B/K.** The iso ratio of policy + allocation peaks at B/K ≈ 2 (offline 1.16–1.18, pool 1.12–1.13) and is smaller
  at B/K 1.25 (1.05): at very tight budgets every policy is constrained by the K pages that must be resident.
- **In bytes and memory.** 10–50 MiB per sequence is small for one request but scales with the batch: at batch 32 and B=128 on 8B,
  the offline allocation + EMA saves 1.6 GiB of cache for the same misses.
- **N1 restated:** "a better policy is worth a few percent of capacity, a better allocation more; both are free at run time" —
  capacity still dominates the absolute miss rate (B 96 → 128 removes 41 % of LRU's misses), but the policy and allocation gains
  apply at every capacity.

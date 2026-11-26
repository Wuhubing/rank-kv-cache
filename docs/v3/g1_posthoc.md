# G1 (post hoc, exploratory): P3's capacity numbers on the evaluated held-out sets

**Conclusion: the iso-miss capacity of policy + allocation on all three held-out sets matches dev to within 0.01–0.02 of the ratio,
so P3's memory figures carry over.** At B=96: offline allocation + EMA 1.083 (0.6B v1), 1.087 (0.6B v3), 1.107 (8B), 1.110 (8B v3)
against dev 1.084 / 1.101; C1 1.061–1.070 against dev 1.061–1.067; EMA 1.053–1.060 against 1.052–1.058; the global pool on v3
1.086 / 1.104 against dev 1.082 / 1.096.

**Post hoc, exploratory.** These replays were run after the three held-out sets had been evaluated, with configurations fixed
beforehand (EMA and allocation + EMA: the preregistered evaluation tables with their frozen α and allocation; C1: γ = 1 frozen in
F2; pool: F3's frozen configuration). The LRU fine grid (64–1024, step 8) on the held-out traces and the C1 replays on the two
earlier held-out sets are new. Nothing was selected from them.

Sources: `analysis_v3.run lru_fine` / `g1` on `data/heldout` (0.6B, longctx_v1), `data/8b_heldout`, `data/heldout_v3_06b`,
`data/heldout_v3_8b`; `runs/run_g1_summary.sh` → `results_v3/g1/<set>/iso.csv`; audits of the g1 replays 2 352 / 3 024 / 2 352 /
3 024 streams, 0 mismatches.

| Set, B | EMA | C1 power law | global pool | offline allocation + EMA |
|---|---|---|---|---|
| 0.6B longctx_v1 held-out, 96 | 1.058 · 10 MiB | 1.067 · 11 | – | 1.083 · 14 |
| 0.6B longctx_v1 held-out, 128 | 1.071 · 16 | 1.090 · 20 | – | 1.155 · 35 |
| 0.6B v3 held-out, 96 | 1.060 · 10 | 1.070 · 12 | 1.086 · 14 | 1.087 · 15 |
| 0.6B v3 held-out, 128 | 1.073 · 16 | 1.094 · 21 | 1.137 · 31 | 1.159 · 36 |
| 8B held-out, 96 | 1.053 · 11 | 1.061 · 13 | – | 1.107 · 23 |
| 8B held-out, 128 | 1.062 · 18 | 1.078 · 23 | – | 1.185 · 53 |
| 8B v3 held-out, 96 | 1.054 · 12 | 1.062 · 13 | 1.104 · 23 | 1.110 · 24 |
| 8B v3 held-out, 128 | 1.065 · 19 | 1.081 · 23 | 1.141 · 41 | 1.193 · 56 |
| (dev, for comparison: 0.6B 96 / 128) | 1.058 / 1.070 | 1.067 / 1.090 | 1.082 / 1.130 | 1.084 / 1.155 |
| (dev: 8B 96 / 128) | 1.052 / 1.061 | 1.061 / 1.078 | 1.096 / 1.124 | 1.101 / 1.175 |

Cells: iso-miss capacity ratio · GPU cache saved per sequence at equal misses (MiB, 64 KiB per page and layer).

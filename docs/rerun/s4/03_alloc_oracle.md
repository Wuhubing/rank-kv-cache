# S4.3 Per-layer allocation with same-allocation Bélády and LRU (dev)

**Conclusion.** Per-layer allocation and score EMA **add up**. On dev at B=96, LODO allocation + EMA has 14.9 % fewer
misses than uniform EMA (997 098 vs 1 049 018; G 0.402 against uniform anchors). Against the Bélády that uses the
**same** per-layer budgets, allocation + hybrid/EMA closes 33 % of the gap (not 37–40 %). Within the allocation (LRU and
Bélády both at the allocated budgets), the replacement part closes 25 % (hybrid) / 28 % (EMA, held-out, S5), about the
same share as at a uniform budget. The LODO allocation recovers 89–94 % of the oracle allocation's gain at B = 96–128, with
a per-layer fold standard deviation of 0.5–2 slots.

Sources: `checks.py` (B6/C10 in `results_v2/dev/efficient_dev/checks_report.txt`, `checks_alloc.csv`,
`checks_alloc_methods.csv`) and `validation/alloc_oracle.py dev` (`results_v2/dev/alloc_oracle/`). The dev allocation file
of `efficient_dev.py` uses the best *hybrid* config as the score policy, so the same-allocation anchors on dev are reported
for `alloc+best_hybrid` and `alloc+lru`. The EMA version with same-allocation anchors was run on held-out
(`docs/rerun/s5_replication.md`).

## Uniform vs allocated budgets (dev, total slots 28·B; misses summed over 14 documents)
| B | LRU uniform | LRU alloc (LODO) | EMA uniform | EMA alloc (LODO) | EMA alloc (oracle) | LODO share of oracle gain |
|---|---|---|---|---|---|---|
| 80 | 1 738 889 | 1 723 964 | 1 594 034 | 1 576 772 | 1 572 279 | 79 % |
| 96 | 1 165 712 | 1 113 848 | 1 049 018 | 997 098 | 991 638 | 90 % |
| 128 | 690 267 | 612 490 | 618 328 | 547 970 | 543 509 | 94 % |

(At B=72 and 256 the grid forces the uniform allocation.) Additivity at B=96: G(EMA) 0.279 + G(alloc+LRU) 0.124 = 0.402 =
G(alloc+EMA) 0.402. At B=128 it is slightly sub-additive (0.268 + 0.289 = 0.557 vs 0.529).

## Same-allocation anchors (dev)
| label | B | G vs uniform anchors | G vs same-allocation Bélády | G within the allocation | repeat-miss drop vs uniform LRU |
|---|---|---|---|---|---|
| alloc+LRU | 96 | 0.124 | 0.110 | 0 | 5.1 % |
| alloc+best hybrid | 96 | 0.373 | 0.332 | 0.249 | 15.5 % |
| alloc+LRU | 128 | 0.289 | 0.239 | 0 | 14.6 % |
| alloc+best hybrid | 128 | 0.484 | 0.400 | 0.213 | 24.4 % |

## Allocation method
Greedy and exact knapsack give the same G (0.402 at B=96 with EMA). Gap-proportional allocation is much worse than
uniform (−0.34), so the curve-based allocation matters.

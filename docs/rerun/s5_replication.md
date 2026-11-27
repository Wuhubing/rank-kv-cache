# S5: cross-hardware replication of the held-out evaluation

**Result: the replication succeeds. All three preregistered criteria pass on traces re-collected on an
RTX 6000 Ada, and every number lands within 0.004 of the L4 original.**

## Timing (preregistration before data)
| Event | Time (UTC) | Source |
|---|---|---|
| Preregistration `docs/replication_v2.md` committed **and pushed** | **2026-11-25T06:58:09Z** | commit `86f1166` |
| First held-out trace collection started | **2026-11-25T06:58:24.83Z** | `started_at` in the manifest of the first held-out document (`data/heldout/*.json` on wulab2; committed copy without the generated text: `results_v2/provenance/manifests/heldout.jsonl`) |
| Last held-out trace finished | 2026-11-25T07:16:02.94Z | manifests |
| Evaluation (steps 2–5, run once) | 2026-11-25T07:20–07:22Z | `logs/s5.log` on wulab2 (copy: `results_v2/provenance/logs/s5.log`) |

The push finished before the collection command was started (same shell, run one after the other). The frozen file
`splits/heldout_frozen_v1.json` is unchanged (it was fitted on the L4 dev traces). Nothing from S3–S4 was used.

## Collection
21/21 held-out documents of `splits/longctx_v1.json`, wulab2 card 0 (`GPU-a74391df-5c27-152f-d0c0-6b86f8ea07d3`),
torch 2.6.0+cu124. Stops: 18 EOS, 2 at 1024 tokens, 1 by the 32-gram loop rule. 442–1023 decode steps
(15 686 in total). Raw traces: 1.22 GB in `data/heldout/` on wulab2 (sha256 in each manifest). Sparse and dense
reference generations differ for every document (expected: the sparse attention changes the logits).

## Preregistered criteria (B = 96)
| # | Criterion | Threshold | L4 (2026-11-23) | **RTX 6000 Ada (this run)** | Pass |
|---|---|---|---|---|---|
| (i) | EMA G, 95 % CI lower bound | > 0.15 | 0.281 [0.277, 0.285] | **0.279 [0.275, 0.282]** | yes |
| (ii) | Documents with EMA < LRU | ≥ 18/21 | 21/21 | **21/21** | yes |
| (iii) | Alloc+EMA − EMA, ΔG CI | > 0 | 0.120 [0.113, 0.128] | **0.119 [0.112, 0.127]** | yes |

Source: `results_v2/heldout/criteria.txt` (`recompute_heldout.py` on `results_v2/heldout/eval/misses.csv`).

## Budget curve, side by side (G against uniform-budget LRU and Bélády; L4 / new)
| B (B/K) | LRU miss % | Bélády % | EMA G | Alloc+EMA G† | dsum G | score G | dcount G | linear G |
|---|---|---|---|---|---|---|---|---|
| 72 (1.125) | 11.50 / 11.42 | 8.81 / 8.75 | 0.257 / 0.256 | 0.257 / 0.256 | 0.239 / 0.236 | 0.229 / 0.228 | 0.110 / 0.104 | – |
| 80 (1.25) | 8.88 / 8.82 | 6.19 / 6.15 | 0.276 / 0.275 | 0.307 / 0.306 | 0.257 / 0.252 | 0.249 / 0.248 | 0.069 / 0.059 | – |
| **96 (1.5)** | 5.98 / 5.95 | 3.85 / 3.84 | **0.281 / 0.279** | 0.402 / 0.398 | 0.270 / 0.266 | 0.246 / 0.245 | 0.044 / 0.031 | 0.303 / 0.300 |
| 128 (2) | 3.56 / 3.56 | 2.19 / 2.20 | 0.271 / 0.268 | 0.528 / 0.524 | 0.266 / 0.263 | 0.224 / 0.221 | 0.052 / 0.044 | 0.314 / 0.311 |
| 256 (4) | 1.38 / 1.41 | 0.98 / 1.00 | 0.242 / 0.239 | 0.242 / 0.239 | 0.216 / 0.213 | 0.145 / 0.143 | 0.011 / 0.011 | 0.336 / 0.333 |

† Against uniform-budget anchors, as preregistered; not a share of the gap at that allocation (see the next table).

Repeat-miss drop vs LRU at B=96 and equal total slots: EMA 11.5 % [11.3, 11.7] (L4 11.6 %), Alloc+EMA 16.4 %
[16.0, 16.8] (L4 16.6 %).

Per document (B=96): EMA G 0.264–0.295 (median 0.279), Alloc+EMA 0.369–0.431, current score 0.221–0.264; every
policy except dcount beats LRU on 21/21 documents. By task: gov_report EMA 0.280, qmsum 0.276.

## Allocation with same-allocation anchors (`validation/alloc_oracle.py heldout`; new, was pending)
| B | G vs uniform anchors (as preregistered) | **G vs Bélády at the same allocation** | **G within the allocation** (LRU and Bélády both at the allocated budgets) | repeat-miss drop vs uniform LRU |
|---|---|---|---|---|
| 80 | 0.306 [0.300, 0.313] | 0.295 [0.290, 0.300] | 0.279 [0.276, 0.282] | 10.2 % |
| **96** | 0.398 [0.390, 0.406] | **0.354 [0.349, 0.360]** | **0.280 [0.276, 0.284]** | 16.4 % |
| 128 | 0.524 [0.510, 0.537] | 0.431 [0.422, 0.439] | 0.258 [0.249, 0.265] | 26.2 % |

(At B=72 and 256 the frozen allocation is uniform, so all three G equal EMA's.) Reading: the allocation itself moves
both LRU and Bélády (at B=96, same-allocation LRU has 4.1 % fewer misses than uniform LRU, and same-allocation
Bélády 6.7 % fewer). Against the real optimum at that allocation, Alloc+EMA closes 35 % of the gap, not 40 %. The
replacement part alone (EMA vs LRU at the same allocation) closes 28 %, the same as uniform EMA. So **allocation
and replacement add up**: allocation lowers both anchors, and EMA closes the same share of the remaining gap.

## Checks
- Independent simulator (`replay.py`) vs kvpolicy for LRU and Bélády: 5 880 (document, layer, policy, budget)
  streams, 0 mismatches, Bélády ≤ LRU everywhere, no zero-miss stream (`results_v2/heldout/audit.txt`).
- `heldout.py eval` checked the folder against the split: exactly the 21 held-out rids, 28 layers, K=64.
- float16 score ties (preregistered as reported): on average 73.8 % of eligible pages share their stored float16
  score with at least one other page at the same (layer, step), and 54 % of (layer, step) pairs have a tie at the
  K-th selected score (sampled every 8th step; `results_v2/heldout/ties.csv`; dev: 74.6 % and 54.8 %). Selection
  itself uses float32 scores on the GPU. Only the stored scores the replacement policies read are float16, and ties
  are broken by the LRU key. Kept as is, for comparability with the original.

## Deviations
**Post-hoc analysis (added in followup-v3, R9).** The float16 tie share above was listed in the preregistration as "also
reported", but its command was not among the preregistered evaluation commands; it was written and run after the evaluation
(`results_v2/heldout/ties.csv`, 07:23Z, after steps 2–5 at 07:20–07:22Z). It is a post-hoc, descriptive reading and took no
part in any criterion or decision.

Otherwise none from the preregistered commands. Steps 2–5 were run once, in order, with the arguments as written; step 1
ran once without interruption (no document collected twice). The GPU stayed the same (card 0) throughout.
Notes that are not deviations: the held-out collection ran in parallel with the CPU-only S4 analyses on dev (S4
results were not used, and the frozen file was not touched); after the evaluation, the result files were moved on
wulab2 from `results_v2/heldout/` to the untracked `out/results_v2/heldout/` so that later `git pull`s do not
collide with the committed copies (same bytes, copied to the repository with rsync first).

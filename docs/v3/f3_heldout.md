# F: third held-out set, evaluated once (confirmatory)

**Result: all preregistered hypotheses pass on both models.** On 21 new GovReport documents (none used before), the score EMA
closes 28 % of the LRU–Bélády gap again (0.6B G 0.281 [0.279, 0.283], 8B 0.277 [0.274, 0.280], 21/21 documents each; H1). The
normalised power-law kernel chosen on dev beats the EMA on every document (ΔG +0.048 [0.044, 0.051] on 0.6B, +0.043 [0.040,
0.046] on 8B; H2). The global percentile pool, which uses no offline statistic, beats uniform per-layer budgets with the EMA
(ΔG +0.119 [0.114, 0.125] and +0.233 [0.217, 0.249]; H3) and reaches 96 % (0.6B) and 91 % (8B) of the frozen offline allocation's
gain at B=96 (77 % and 63 % at B=128).

## Timing (preregistration before data)
| Event | Time (UTC) | Source |
|---|---|---|
| `docs/preregistration_v3.md` + `splits/heldout_frozen_v3.json` committed | 2026-11-26T00:15:29Z | commit `087d53f` (GitHub committer date) |
| Push confirmed on GitHub (`git ls-remote` returned `087d53f`) | 2026-11-26T00:15:31Z | session log |
| Collection script started (prints the commit and origin's head, both `087d53f`) | **2026-11-26T00:15:39Z** | `results_v3/provenance/f3_collect.log` |
| First held-out trace started | 2026-11-26T00:15:47Z | manifests (`results_v3/provenance/heldout_v3_06b_manifests.jsonl`) |
| 0.6B done / 8B started / 8B done | 00:26:58 / 00:27:08 / 00:48:22Z | manifests |
| Evaluation (the six preregistered commands, run once) | 01:31:33–01:37:23Z | `results_v3/provenance/f3_eval.log` |

Evaluation ran at commit `c70f94f`; `git diff 087d53f c70f94f` is empty for the evaluation code, the policies, the audit and the
frozen file (later commits only added documents and results).

## Collection
wulab2 card 1 (`GPU-7ee1d08f-1ddf-05b7-9480-8047c2574e74`), `runs/run_f3_collect.sh`, one run, no interruption, no document
collected twice. 0.6B (`splits/longctx_v3.json`): 21/21, stops 19 EOS, 1 loop, 1 at 1024 tokens; 535–1023 decode steps. 8B
(`splits/longctx_v3_8b.json`): 21/21, all at the 1024-token cap; the thinking block ends at step 321–944, so each 8B trace
contains the whole thinking phase and the start of the answer (as for the earlier 8B sets). Raw traces stay on wulab2
(`data/heldout_v3_06b`, `data/heldout_v3_8b`); manifests without generated text are committed.

## Criteria (B = 96 decides; B = 128 secondary; document bootstrap 2000, seed 0)
| | 0.6B B=96 | 0.6B B=128 | 8B B=96 | 8B B=128 |
|---|---|---|---|---|
| **H1** EMA G [CI], documents better than LRU (threshold: lower bound > 0.20, ≥ 18/21) | **0.281 [0.279, 0.283], 21/21 — pass** | 0.274 [0.269, 0.278], 21/21 | **0.277 [0.274, 0.280], 21/21 — pass** | 0.267 [0.263, 0.272], 21/21 |
| P2 normalised power law, G | 0.328 [0.324, 0.333] | 0.340 | 0.320 [0.315, 0.325] | 0.322 |
| **H2** P2 − EMA, ΔG [CI], documents (threshold: lower bound > 0) | **+0.048 [0.044, 0.051], 21/21 — pass** | +0.066 [0.063, 0.069], 21/21 | **+0.043 [0.040, 0.046], 21/21 — pass** | +0.055 [0.050, 0.059], 21/21 |
| P4 global percentile pool, G | 0.400 [0.394, 0.406] | 0.474 | 0.510 [0.492, 0.528] | 0.520 |
| **H3** P4 − uniform EMA, ΔG [CI], documents (threshold: lower bound > 0) | **+0.119 [0.114, 0.125], 21/21 — pass** | +0.200 [0.192, 0.208], 21/21 | **+0.233 [0.217, 0.249], 21/21 — pass** | +0.253 [0.239, 0.268], 21/21 |
| P4 share of the offline allocation's gain (reported, no threshold) | 0.96 | 0.76 | 0.91 | 0.63 |
| Offline allocation + EMA (frozen), G against uniform anchors | 0.406 | 0.535 | 0.534 | 0.669 |
| Offline allocation + EMA, G within its own allocation | 0.284 | 0.260 | 0.276 | 0.248 |

Sources: `results_v3/f3/{06b,8b}/criteria.txt` / `criteria.json` (`analysis_v3/f_eval.py`), miss tables `misses.csv.gz`. Audit:
LRU and Bélády of every (document, layer, budget) recomputed with `replay.py`: 2 352 (0.6B) and 3 024 (8B) streams, 0
mismatches, 0 missing rows (`audit.txt`).

## What is now confirmatory and what is not
- Confirmatory: the EMA's G on a third document set (third replication on 0.6B, second on 8B); the normalised power-law kernel
  beating the EMA (P2); the global percentile pool beating uniform budgets without offline curves (P4, H3).
- Not claimed: that the pool matches the offline allocation (it reaches 91–96 % of its gain at B=96 and 63–77 % at B=128, reported
  without a threshold; on dev it was also below the offline allocation at B=128).
- The three held-out sets are now all used; none of them may be used for further selection.

## Deviations
None from the preregistered commands. Process notes: the C/D CPU queue was paused at a job boundary during the evaluation so the
task stayed within the 32-process limit (no effect on results); the C4 dev replays that chose the P4 configuration ran on the
mini (before the user's decision to run all computation on wulab2), from copies of the same dev traces.

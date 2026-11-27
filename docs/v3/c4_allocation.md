# C4: allocation without offline curves (online reallocation and a global pool)

**Conclusion: a single global pool of slots, evicting the page with the lowest percentile-EMA across all layers, recovers most of
the offline per-layer allocation's gain at B=96 without any offline statistic, on both models; at B=128 it recovers only
60–74 %.** At B=96 the pool reaches 95 % [88, 101] (0.6B) and 93 % [82, 102] (8B) of the offline LODO allocation's gain over
uniform budgets; against uniform budgets + EMA it gains ΔG +0.117 [0.111, 0.123] (0.6B) and +0.216 [0.190, 0.240] (8B), on
14/14 documents. The ghost-list online reallocator reaches 79 % on both models at B=96 and 56–59 % at B=128. It passes the C
gate (against uniform + EMA) and is the P4 candidate for F; it does not beat the offline allocation (B=96 ΔG −0.007 [−0.015,
0.001] and −0.016 [−0.042, 0.005]). At B=256 the per-layer grid cannot give any layer more than 256 slots, so the offline
allocation is uniform there, while the pool (and even a global LRU) moves slots freely and gains a lot (G 0.58–0.82 against
uniform anchors); at B=80 a global LRU is far worse than per-layer LRU (−0.40).

Source: `analysis_v3/c4_alloc.py` (curves, online, pool; tests `tests/test_c4_alloc.py`: without moves the online policy equals the
per-layer EMA; budgets stay at the total and ≥ K; the pool's global Bélády equals per-layer Bélády for one layer). Run on the
mini on copies of the dev traces (before the 2026-11-25 decision to run all computation on wulab2), results committed in
`results_v3/c4/{dev,8b}/` (`summary.csv`, `shares.csv`, `rows.csv.gz`). Anchors: G against uniform-budget LRU and Bélády; the
offline allocation is `kvpolicy.allocate` (greedy) on the summed EMA curves of the other 13 documents (grid 64–256), "oracle" on
the document's own curves; same-allocation LRU/Bélády are sums of per-layer curve entries. Online: N ∈ {16, 64, 256} steps,
s ∈ {2, 4, 8} slots, decay ∈ {0, 0.5}; pool: signal z or percentile, β ∈ {0, 0.5, 2}; both tuned LODO. The pool's own LRU/Bélády
replays are not audited with `replay.py` (they are a different, global cache model); the per-layer curves' LRU/Bélády equal the
audited per-layer replays of B1/B7 at the shared budgets.

## G against uniform-budget anchors (14 dev documents each)
| Policy | 0.6B B=80 | **B=96** | B=128 | B=256 | 8B B=80 | **B=96** | B=128 | B=256 |
|---|---|---|---|---|---|---|---|---|
| uniform budgets + EMA | 0.273 | **0.279** | 0.268 | 0.194 | 0.270 | **0.271** | 0.256 | 0.161 |
| offline allocation (LODO curves) + EMA | 0.305 | 0.402 | 0.529 | (uniform) | 0.365 | 0.504 | 0.631 | (uniform) |
| oracle allocation (own curves) + EMA | 0.314 | 0.415 | 0.546 | (uniform) | 0.376 | 0.521 | 0.655 | (uniform) |
| online ghost reallocation + EMA | 0.313 | 0.377 | 0.421 | 0.330 | 0.366 | 0.455 | 0.467 | 0.306 |
| global pool, EMA of z (+ β miss rate) | 0.272 | 0.346 | 0.444 | 0.742 | 0.361 | 0.457 | 0.482 | 0.724 |
| **global pool, EMA of percentile (+ β miss rate)** | 0.290 | **0.396** | 0.461 | 0.577 | 0.362 | **0.487** | 0.480 | 0.696 |
| global pool, LRU | −0.395 | −0.007 | 0.301 | 0.825 | −0.281 | 0.166 | 0.459 | 0.884 |
| global pool, Bélády | 1.450 | 1.329 | 1.328 | 1.372 | 1.541 | 1.435 | 1.416 | 1.400 |

## Share of the offline allocation's gain reached without offline curves, and the C gate
| | 0.6B B=96 | 0.6B B=128 | 8B B=96 | 8B B=128 |
|---|---|---|---|---|
| online: share [CI] | 0.79 [0.77, 0.82] | 0.59 | 0.79 [0.77, 0.82] | 0.56 |
| pool (percentile): share [CI] | **0.95 [0.88, 1.01]** | 0.74 | **0.93 [0.82, 1.02]** | 0.60 |
| pool (percentile) − uniform EMA, ΔG [CI] | **+0.117 [0.111, 0.123]** | +0.194 | **+0.216 [0.190, 0.240]** | +0.224 |
| pool (percentile) − offline allocation, ΔG [CI] | −0.007 [−0.015, 0.001] | −0.068 | −0.016 [−0.042, 0.005] | −0.151 |
| oracle − offline allocation | +0.013 | +0.017 | +0.017 | +0.024 |

Gate (task section 7, written as "against uniform budgets + EMA"): CI lower bound > 0 at B=96 on both models, point estimate ≥ 0
at B=128: passed. The "≥ 80 % of the offline allocation's gain" support condition of P4 holds at B=96 on both models and fails at
B=128.

## Reading
- The percentile signal is what makes pages of different layers comparable: z is relative to each layer's own top-1/K-th spread,
  and pooling on z is worse than pooling on the percentile at B ≤ 128.
- The pool's advantage at large B and its weakness at small B have one cause: with a global victim choice, layers with many
  candidates near the selection cut lose slots to layers with stale pages. At B=80 the global LRU starves layers and is far worse
  than per-layer LRU; the score signal fixes most of that.
- Online reallocation is limited by the ghost signal (only s recently evicted pages) and by the slow, one-step-at-a-time moves;
  at B=128 the offline curves' knowledge of the whole budget range is worth much more.
- P4 is therefore "the global percentile pool reaches the offline allocation at tight budgets and beats uniform budgets at every
  budget, without offline statistics" (dev, both models), with the B=128 shortfall stated.

## Exact pool semantics
Added 2026-11-27 (review of PR #11, R11-4): the one-line description above does not determine the implementation; these are the
rules of `analysis_v3/c4_alloc.py: pool_run`, the same text as the dated appendix of `docs/preregistration_v3.md` (clarification
only; frozen configuration and code unchanged). `tests/test_pool_spec.py` re-implements the pool from this text alone and matches
`pool_run` miss for miss on synthetic traces (LRU, Bélády, EMA with both signals, β ∈ {0, 0.5, 2}, α ∈ {0.3, 0.5}, incl. layers with
fewer than K pages and tied scores). The reviewer notes that a "serve all layers of a step at once" variant would have given the
frozen configurations 0.9–1.9 % fewer misses on the review slice, i.e. the implemented semantics is, if anything, conservative for
the pool.

0. Input per decode step t and layer l: the pages layer l selects (the selector's top-K, in descending score order; fewer
   than K while the layer has fewer than K pages) and the score of every page of layer l (non-finite = not an eligible page).
   The pool has L·B slots shared by all layers; a slot holds one (layer, page).
1. Layers are served one at a time within a decode step, in order 0 … L−1; global time u = t·L + l. While layer l is served,
   only layer l's current selection is protected; pages of other layers (including layers later in the same step) are eviction
   candidates. If layer l's non-resident selected pages do not fit, n = (resident pages + those misses − L·B) victims are
   evicted at once: the n candidates with the smallest key (rules 4 and 5). Every selected page then becomes resident with
   last use u.
2. The layer's running miss rate is m_l ← 0.99·m_l + 0.01·(misses of layer l at this step / number of pages layer l selects
   at this step, which is K once the layer has at least K pages), updated after layer l is served; m_l starts at 0. A victim
   key evaluated while layer l is served uses each layer's current m (for layer l: its value before this step's update).
3. The signal is the mid-rank percentile of the page's current score among all pages of that layer with a finite score at that
   step (`kvpolicy._allrank`): (#{q: s_q < s} + (#{q: s_q = s} − 1) / 2) / (N − 1); 1 if N = 1; 0 for a non-finite score.
   At each service of layer l, before any eviction, every resident page of layer l (hits included) is updated
   v ← α·v + (1−α)·signal; a newly admitted page starts at v = its current signal (no further update in that service).
   (The non-frozen signal "z" of the dev grid replaces the percentile by (s − s_K) / (s_1 − s_K + 10⁻⁶), s_1 and s_K being the
   scores of the first and the last selected page.)
4. Victim = the candidate with the smallest key (v + β·m_l, last use u, layer, page id), where l is the candidate's layer;
   i.e. ties go to the least recently used, then the lower layer, then the lower page id.
5. The global-pool LRU baseline uses key (last use u, layer, page id), without the "was a miss" tie-break of the per-layer LRU,
   and evicts the lower page id first (per-layer LRU evicts the higher id first). The G anchors are the per-layer LRU/Bélády and
   are not affected. The global Bélády uses (−next use in global time, layer, page id); a page with no further use has next use
   +∞ and leaves first.

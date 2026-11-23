# Related Work Matrix (Phase A)

**Question for Gate A.** Has prior work already used a page's *historical* sparse-attention selection rank to decide which pages stay in a recallable GPU KV cache, while the sparse selection itself stays fixed? And has anyone measured whether that rank predicts reuse beyond recency and frequency?

**How this was checked.** Every claim below cites the section, figure or table it came from. Quotes are under 25 words.

*First pass (2026-11-22, morning):*
- **Read from the full PDF text and spot-checked by `grep`:** ArkVale (NeurIPS 2024 proceedings PDF) and HiSparse (arXiv:2608.07009v1).
- **Read from code:** ArkVale's eviction kernel, `source/arkvale_cpp/src/select.cu` in `github.com/pku-liang/ArkVale`.
- **Read via the arXiv HTML and spot-checked against PDF text:** Quest (arXiv:2406.10774), H₂O (arXiv:2306.14048) and PARROT (arXiv:2006.16239), for orientation only.

*Second pass (2026-11-22, afternoon), closing the items the first pass left unverified:*
- **InfLLM** (arXiv:2402.04617): PDF text, §3.2, §4.2, Appendix A and C.1. Experiment tables skimmed only.
- **Scissorhands** (arXiv:2305.17118): ar5iv HTML, about 98% of the text (the unread tail is references or appendix).
- **H₂O appendices:** ar5iv HTML up to the end of Appendix C and most of Appendix D (the last ~7% of the text was not read).
- **HiSparse code:** upstream SGLang `python/sglang/kernels/jit/csrc/kvcacheio/hisparse.cuh`, plus a re-check of §3.2, §3.4, §4.1 and Fig. 6.
- **Quest code:** `mit-han-lab/Quest`, `evaluation/quest_attention.py` and the evaluation scripts.
- **PARROT:** arXiv PDF, §4.4, §5.1–5.2, Table 1, Fig. 6.
- **ArkVale OpenReview** (forum id `4oAt5L4lYe`): **could not be read**. Both the page and the API returned a bot-check challenge, which we did not try to bypass.

The Scissorhands and H₂O quotes were extracted through a page-reading tool rather than from the raw PDF, so spot-check them against the PDF before quoting them in the report.

## Matrix

| Paper | Claim | Method | Evidence | Assumptions | Overlap with H1 / H2 | Gap |
|---|---|---|---|---|---|---|
| **ArkVale** (Chen et al., NeurIPS 2024) | Page importance changes over decoding, so evicted pages must be recallable. Ranking pages with digests recovers accuracy at small budgets. | Filled pages are backed up to CPU memory and a key digest stays on the GPU. Every step, the *current query* scores all pages (resident or evicted). Top-k pages are attended; missing ones are recalled and swapped with the *bottom-ranked* resident pages. The cuboid-mean digest is a per-dimension bound. | §4.2 / Fig. 4a: rank trajectories of 7 pages from one GovReport sample. Fig. 4b: offline simulation of recalls per step ("under 10" per step when k < min(40, c/2)). §5.1 / Fig. 5: the algorithm. §5.2, Eqs. 1–4: digests. §6, Figs. 7–9 and Table 1: accuracy, latency, throughput. "Recall accuracy" in Fig. 7 measures the importance estimator, not cache hits. | LongChat-7b-v1.5-32k on one A100 80GB, 32-token pages. k ≈ c/2, so attended pages are about half the resident slots. Baselines are StreamingLLM, H₂O and TOVA, not Quest. | **H1:** Fig. 4a shows importance drifts, but only anecdotally, with no aggregate reuse statistic. **H2:** it is a recallable host-backed page cache, the same structure as ours. | Victims are chosen by the **current** step's score only; the kernel has no recency, frequency or history state. Selection and eviction share one score. The camera-ready has no eviction ablation (no LRU vs. score), no oracle and no miss-rate metric (searched for evict, LRU, miss, oracle, history, recen, ablat). |
| **InfLLM** (Xiao et al., NeurIPS 2024) | Training-free long-context extrapolation with a block-level context memory. Most memory units live in CPU memory, and the GPU cache miss rate "is quite low" (§3.2). | 128-token memory blocks, each represented by its 4 highest-scoring tokens (§4.2). Relevance to representative tokens picks the blocks to attend. A GPU cache of 32 units ("twice the number of loaded units", App. C.1) holds blocks; misses are copied from CPU (App. A). Eviction sorts units by a decayed score s_b = s_b·d + Σ attention scores of current queries on the unit's keys, d = 0.1, and offloads the lowest (App. A, Eq. 1). | App. A / Fig. 4: miss rate (roughly 3–5%) against score decay on "a sample of data from the GovReport dataset", lower than Random and FIFO. No table of numbers. | Block-level, not page-level. The cache score uses post-attention mass, not the block-selection score or rank. | **H2 (closest caching policy):** a host-backed, recallable block cache whose replacement is an attention-weighted, exponentially decayed frequency score, compared against FIFO and Random on GovReport. The paper calls it "an LRU strategy" although the rule is a decayed score. | No oracle/Bélády comparison and no comparison with recency-plus-frequency hybrids. It never uses the selection score or top-K rank for caching, and never tests whether a past-importance signal adds information beyond recency and frequency. |
| **HiSparse** (Xie et al., arXiv:2608.07009, 2026) | Exact sparse-attention decoding with full KV in host memory and a fixed B-slot GPU cache per request and layer. LRU tracks Bélády and is a good online proxy. | Per request-layer LRU. "Within a step, entries that hit are promoted above the newly fetched misses in the recency order" (§3.2); misses are placed "just behind the hits" (§3.4). A fused kernel marks slots hit or evictable, scans for victims and fetches misses (§3.4). | §4.3 / Fig. 6 (all re-confirmed): one GLM-5.1 request with a 100,384-token LongBenchV2 prompt, 1,799 decode steps, 78 sparse layers, k=2048. B=k misses 30%; at B=4096, LRU 13.4%, FIFO 17.2%, random 16.1%, Bélády 8.2%; LRU at B=8192 6.7%. §4.4 / Fig. 7: miss-resolution kernel cost against B. | **Token-granular cache.** The paper is ambiguous for Quest (Table 1 calls it page-level, but every selector "emits logical token positions" and §3.2 maps "each layer-local logical position" to a slot). The code is explicit: residency and LRU "remain token-granular even when the sparse-attention selection arrives as block ids". Qwen3-30B-A3B+Quest appears only in throughput experiments (§4.1–4.2). | **H2:** nearly identical setup (fixed selection, host KV, B-slot cache, FIFO/random/LRU/Bélády replay). Its hit-over-fill rule is already a binary "selected again" signal. | Selection score and rank are never used for replacement; frequency is not studied. The Bélády result bounds "what a smarter, predictive replacement policy could still recover" (§4.3). The paper states no victim tie-break; in the code the LRU state is a total order, victims are the oldest evictable slots, and a step's misses enter in top-k array order (so the miss with the smallest selection index becomes oldest). This is current upstream code, not necessarily the evaluated version. |
| **Quest** (Tang et al., ICML 2024) | Query-aware page selection approximates full attention while reading a fraction of the KV. | Per-page channel min/max keys. The upper bound Σᵢ max(qᵢmᵢ, qᵢMᵢ) ranks pages and the top pages are attended (§3.4, Alg. 1). Page size 16 in the kernels (§4.3.1) and in all evaluation scripts (`--chunk_size 16`); Fig. 8 sweeps it. | §3 and Alg. 1; page-size discussion in §4. | Quest "retains all of the KV cache" (§2.2) and "does not discard KV cache" (§4.2.2). GPU residency is implied, never stated. **Selection is per query head** in the code (`repeat_kv`, then `topk` per head); the paper does not discuss heads or GQA, and its models are multi-head, not GQA. | **Neither H1 nor H2.** It is the selector we adapt. **Our selector differs:** we average the bounds over query heads so that all heads of a layer share one ranking and one cache. | No analysis of how selected pages are reused across decode steps, and no memory hierarchy. Its ranking is never used for residency. |
| **H₂O** (Zhang et al., NeurIPS 2023) | A few heavy-hitter tokens dominate attention. Keeping heavy hitters plus recent tokens preserves quality at a small budget. | Accumulated attention score plus a recent window. Token-level and **permanent** eviction: "the evicted KV will not be seen in the future steps" (§5.1); at most one KV evicted per step (Def. 4.3). | Accumulated attention follows a power law (§3.2). Ablation (§5.3, App. C.6 / Table 9): heavy hitters alone or recent tokens alone fail, while "Incorporating both components" recovers accuracy. | OPT and LLaMA-family models. A lossy cache: eviction changes what is attended. | **H2 (form):** an importance-plus-recency retention score, which is the shape of our s_i. | Importance is cumulative attention, not the last top-K rank. Nothing is recallable. Bélády is called "not applicable for KV cache design" (§3.2); no oracle experiment and no miss-rate figure despite the claim of a lower miss rate (§1). **Heavy-hitter stability across decoding steps is never measured:** App. C.8 / Fig. 10 is a static per-layer distribution, and App. C.10 / Fig. 13 tracks changes across training checkpoints, not decode steps. |
| **Scissorhands** (Liu et al., NeurIPS 2023) | "Persistence of importance": only tokens that had substantial influence at an earlier step matter at a later step (§3.2). | A token is pivotal at step t if its attention exceeds 1/t. A counter over a history window (w=400) counts how often each token's attention falls below that threshold, and the highest-count tokens are dropped, while the r=10 most recent tokens are always kept (§4.1, Alg. 2). Eviction is permanent ("drops stored but non-influential tokens"). | §3.2 / Fig. 2: the persistence ratio (overlap of pivotal sets between the first and second half of a sequence) is "generally over 95%" (§1 says "over 90%"). Accuracy and perplexity vs. compression (Fig. 3); Theorem 4.1. | Token-level, permanent eviction. LRU is cited only as inspiration (§4.1). | **H1 (closest claim):** asserts that past importance predicts future importance. | The evidence is one coarse set overlap (first half vs. second half), not a per-step predictor, and it is not controlled for recency or frequency. No recallable cache, no LRU or Bélády experiment, no miss rates. |
| **PARROT** (Liu et al., ICML 2020) | Imitating Bélády yields a learned replacement policy that beats hand-designed ones. | LSTM over the access history, attention over resident lines, DAgger and a ranking loss. Predicting reuse distance as an auxiliary task (§4.4) gives "16.8% higher normalized cache hit rates" than ablating it (Fig. 6). | Normalized hit rate (r−r_LRU)/(r_opt−r_LRU), with LRU = 0 and Bélády = 1 (§5.1). Table 1, SPEC2006 16-way 2 MB LLC: "PARROT achieves 16% higher cache hit rates than LRU" (§5.2). This is the **raw** hit rate; the paper does not say whether it is relative or percentage points. | CPU last-level cache traces. Model latency far exceeds a per-step GPU kernel budget. | **H1 (general form):** predicting reuse distance from history. **H2:** methodological precedent for normalizing against LRU and Bélády, as our G(π) does. | No model-native signal like attention rank. Its gains do not transfer to KV caching. |

## Gate A conclusion

**ArkVale does not use historical rank for eviction.**
- §5.1 says that "prior to attention computation" every page is scored "using the query and page digests". Missing top-k pages are then recalled and "bottom-ranked pages in GPU will be evicted for exchange".
- The kernel confirms this. In `select.cu`, `new_in[i] = in[pos_ids[i]]` (line 127) copies the resident slots' scores from the **current** step. Line 131 then takes the `n_recalls` lowest of them as victims. No recency, frequency or past-rank state exists.
- ArkVale's §2 presents history-based eviction as a weakness of other methods: those methods "rely on historical data to dictate cache eviction".

**InfLLM uses past attention, but not selection rank, for caching.** Its replacement score accumulates decayed post-attention mass over past steps. It does not use the block-selection score or rank, never compares against an oracle, and never asks whether past importance adds information beyond recency and frequency.

**Scissorhands claims persistence but does not test it as a predictor.** Its only measurement is a coarse first-half vs. second-half overlap, with no control for recency or frequency.

**No paper we read uses the last observed top-K rank as a residency signal with selection held fixed, or tests whether it predicts reuse beyond recency and frequency.**
- HiSparse is the closest setup (same cache model and baselines, pure LRU, the LRU–Bélády gap left open). InfLLM is the closest caching policy (an attention-weighted decayed frequency score).

**Gate A passes, with a narrower framing than "rank-aware eviction is new".** The open question we can claim is incremental: given fixed Quest-style selection and a recallable page cache, does the *last observed* selection rank carry reuse information beyond recency and frequency (H1)? And does adding it to a recency-plus-frequency policy close part of the LRU–Bélády gap (H2)?

## What we can claim

- We hold sparse selection fixed and study only residency, so every selected page stays retrievable and attention is unchanged. ArkVale ties eviction to the current ranking; we decouple the two.
- To our knowledge, based on ArkVale, HiSparse, InfLLM, Quest, H₂O, Scissorhands and PARROT, none tests whether historical top-K selection rank predicts future selection beyond recency and frequency.
- HiSparse reports a 5.2-point LRU–Bélády gap at B=2k on a token-level GLM-5.1 trace and calls it the room for a predictive policy (§4.3). We measure the corresponding gap for page-level Quest-style selection on Qwen3-0.6B.
- We use PARROT-style normalization against LRU and Bélády for G(π).

## Claims we must avoid

- That we are the first to use attention importance for KV eviction. H₂O, Scissorhands, ArkVale and TOVA already do.
- That we are the first to use past importance for **recallable** cache replacement. InfLLM's GPU cache already ranks units by decayed past attention mass.
- That we are the first host-backed or recallable KV cache. ArkVale, InfLLM and HiSparse are.
- That we are the first to compare replacement policies or report miss rates on sparse-attention traces. HiSparse Fig. 6 compares FIFO, random, LRU and Bélády; InfLLM App. A compares its policy with FIFO and Random on GovReport. Our FIFO, LRU and Bélády rows replicate HiSparse's methodology at page granularity.
- Any direct comparison of our miss rates with HiSparse's 13.4% and 8.2% or InfLLM's 3–5%. Theirs are token-level GLM-5.1 with k=2048, and 128-token blocks on different models; ours are 16-token pages, Qwen3-0.6B, K=64.
- That HiSparse's Fig. 6 covers Quest or Qwen3. It does not; Qwen3+Quest appears only in throughput results, and HiSparse caches tokens, not pages.
- That our selector is exactly Quest. Quest selects per query head; we average bounds over query heads so one ranking and one cache serve each layer. Call it "Quest-style".
- That ArkVale measured how often evicted pages return. Fig. 4a shows a single anecdote and Fig. 4b is an offline recall-count simulation.
- That H₂O or Scissorhands measured importance persistence per decoding step. H₂O does not measure it over decode steps; Scissorhands reports one coarse overlap ratio.
- That PARROT's 16% is a normalized gain, or that its gains carry over to KV caches. The 16% is a raw hit-rate improvement on SPEC2006.

## Consequences for the plan

1. **A current-score baseline.** ArkVale's eviction rule ("evict the resident page with the lowest current-step score") is a natural extra baseline between LRU and our policy. It needs the current scores of unselected resident pages, which trace v1 does not record.
2. **An InfLLM-style baseline.** A decayed attention-mass score (d = 0.1) is the closest published caching policy. It needs post-attention mass per page, which trace v1 does not record either. Our learned recency-plus-frequency model is the nearest proxy we can replay today.
3. **The H1 framing.** H1 should test rank against recency *and* frequency, as the blueprint says. HiSparse's hit-over-fill promotion already captures "selected repeatedly", so rank must add information beyond that.
4. **Selector deviation.** Report that our head-averaged bound differs from Quest's per-head selection. A per-head selector would need one cache per head or a union of per-head selections.
5. **Headroom first.** If the page-level Qwen3 LRU–Bélády gap at B=128 is much smaller than HiSparse's token-level gap, report that as a finding before tuning δ and β.

## Still unverified

- **ArkVale OpenReview reviews and rebuttal:** blocked by a bot check. Someone should open forum `4oAt5L4lYe` in a normal browser and check whether reviewers asked about eviction ablations or history-based eviction.
- **HiSparse:** whether the code read (current upstream SGLang) matches the version evaluated in the paper, and whether its Quest path uses the block-to-token expansion (`TopKIsBlocks`).
- **H₂O:** the last ~7% of the text (end of Appendix D).
- **Scissorhands:** the last ~2% of the text (likely references or appendix).
- **Quote extraction:** the Scissorhands and H₂O quotes came through a page-reading tool; spot-check them against the PDF before quoting.
- **TOVA:** cited by ArkVale as a baseline but not read by us.

## Policy baselines from related work (efficient-policy study, 2026-11-23)

Sources forwarded by the coordinating session. "Method read" means its subagent read the method
section; the other entries are abstract-level and must be checked before they are cited.

| Work | Replacement / allocation rule | Status | Our replay (dev, B=96, LODO G vs LRU; LRU 6.03%, Bélády 3.86%) |
|---|---|---|---|
| **InfLLM** (arXiv:2402.04617, App. A) | s_b ← d·s_b + Σ attention mass on the unit, d=0.1; evict the lowest s_b | method read | Decayed sum over **every resident**, with softmax of the Quest bounds over residents: G +0.269 (d=0.5, LODO) and +0.264 at InfLLM's d=0.1. Updating only on selection (the earlier proxy) gives G +0.028. |
| **KVDrive** (arXiv:2605.18071, §5) | (1) evict the lowest **current-step** attention score, ported to Quest/ShadowKV/RetroInfer and compared with LRU; (2) offline profiling plus a multiple-choice knapsack (greedy) for per-layer×head windows under a total GPU budget | method read | (1) The current-score policy gives G +0.244. (2) An exact knapsack on per-layer miss curves gives the same allocation as our greedy (G with EMA 0.409 vs 0.409; with LRU 0.127 vs 0.127). |
| **HiSparse** (arXiv:2608.07009) | token-granular LRU; hits promoted above newly fetched misses within a step | method read | Our default LRU tie-break is the same "hits above misses" order. The alternatives are worse: misses kept above hits gives G −0.044, page-ID ties −0.029. HiSparse reports little residual value in cross-layer coordination, whereas our per-layer allocation adds G +0.127 (LRU) and +0.13 on top of EMA at B=96 (see below). |
| **MoE expert caching** (arXiv:2611.15842) | a decayed usage count closes about 27% of the gap; learned scoring is no better; evicting by the current router logit moves 46% more than LRU | abstract-level | A decayed selection count with no score gives G +0.044 (d=0.8). Scores matter much more here than in the MoE setting. Our current-score policy beats LRU (G +0.244), unlike their current-logit result. |
| RetroInfer / PQCache | LRU best among tried; LFU ≈ LRU | abstract-level | Consistent with our results: LFU −0.84, LRU-2 −0.05, ARC −0.01. |

**How our score EMA relates to InfLLM.**
- Our rule is e ← α·e + (1−α)·z over every resident, where z is the Quest bound normalised by the best and K-th selected scores.
- InfLLM's rule is s ← d·s + a, where a is attention mass.
- With the same update set and the same input the two differ only by the factor (1−α), which does not change the order apart from how a newly admitted page is initialised.
- Measured: InfLLM-style softmax over residents with d=0.5 gives G 0.269, and our EMA with α=0.5 gives 0.279. They agree on every budget to within 0.01–0.04, and the decay sweeps have the same shape.
- **Our policy should be described as an InfLLM-style decayed score rule applied to Quest upper-bound scores of every resident page**, not as a new rule.
- What matters, measured:
  - (a) **Update every resident with its current score.** Updating only selected pages gives G ≤ 0.03, or below LRU.
  - (b) **Normalise or bound the per-step value.** A raw-score sum with d ≥ 0.5 is worse than LRU (G −0.04 to −0.50), because unnormalised sums let a page's age dominate.
  - (c) **Choose d by budget.** The best d is 0.1–0.5 at B/K ≤ 2 and higher at B/K = 4.

**Allocation and HiSparse's "little cross-layer value".** Possible reasons for the disagreement, none tested:
- (i) Granularity: we cache 16-token pages.
- (ii) Budget: at B/K = 4 our allocation gain also vanishes, and HiSparse's headline is B/K = 2 at token level.
- (iii) Model and selector: our gap is concentrated in a few layers (11, 16, 24, 25).
- (iv) Allocation criterion: per-layer miss curves. Budgets proportional to the per-layer LRU–Bélády gap are much **worse** than uniform (G −0.32 with EMA at B=96), because layers with a small gap are pushed down to K slots.

**Claims we must not make (in addition to those above):**
- Using attention or selector scores for recallable-cache replacement is not new (InfLLM, ArkVale, KVDrive).
- Decayed accumulated-score replacement is not new (InfLLM).
- Per-layer GPU budget allocation in a recallable cache is not new (KVDrive).
- "Frequency-style policies are no better than LRU" is not new (RetroInfer, PQCache).

**What we can claim (dev so far):**
- A systematic comparison, under Quest upper-bound scores and a page-level recallable cache, of score-aware, frequency and learned (non-RL) policies, measured as the share of the Bélády gap closed, with LODO tuning and document-clustered CIs.
- A decay-rate sweep showing the best decay rises with budget.
- Ablations that isolate the needed information: the current scores of unselected residents. Rank-only information, as in trace v1, recovers almost nothing.
- The gap is concentrated in a few layers. Allocation by per-layer miss curves (equivalent to an exact knapsack) adds to the score rule. This is set against KVDrive's transfer-benefit criterion and HiSparse's "little cross-layer value".

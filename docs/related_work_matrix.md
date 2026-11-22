# Related Work Matrix (Phase A)

**Question for Gate A.** Has prior work already used a page's *historical* sparse-attention selection rank to decide which pages stay in a recallable GPU KV cache, while the sparse selection itself stays fixed? And has anyone measured whether that rank predicts reuse beyond recency and frequency?

**How this was checked (2026-11-22).** Every claim below cites the section, figure or table it came from. Quotes are under 25 words.
- **Read from the full PDF text and spot-checked by `grep`:** ArkVale (NeurIPS 2024 proceedings PDF) and HiSparse (arXiv:2608.07009v1).
- **Read from code:** ArkVale's eviction kernel, `source/arkvale_cpp/src/select.cu` in `github.com/pku-liang/ArkVale`.
- **Read via the arXiv HTML and spot-checked against PDF text:** Quest (arXiv:2406.10774), H₂O (arXiv:2306.14048) and PARROT (arXiv:2006.16239). These were read for orientation only, so their rows are less detailed.
- **Not read at all:** Scissorhands, InfLLM, and the OpenReview discussions. Anything about them is marked *unverified*.

## Matrix

| Paper | Claim | Method | Evidence | Assumptions | Overlap with H1 / H2 | Gap |
|---|---|---|---|---|---|---|
| **ArkVale** (Chen et al., NeurIPS 2024) | Page importance changes over decoding, so evicted pages must be recallable. Ranking pages with digests recovers accuracy at small budgets. | Filled pages are backed up to CPU memory and a key digest stays on the GPU. Every step, the *current query* scores all pages (resident or evicted). Top-k pages are attended; missing ones are recalled and swapped with the *bottom-ranked* resident pages. The cuboid-mean digest is a per-dimension bound. | §4.2 / Fig. 4a: rank trajectories of 7 pages from one GovReport sample. Fig. 4b: offline simulation of recalls per step. §5.1 / Fig. 5: the algorithm. §5.2, Eqs. 1–4: digests. §6, Figs. 7–9 and Table 1: accuracy, latency, throughput. | LongChat-7b-v1.5-32k on one A100 80GB, 32-token pages. k ≈ c/2, so attended pages are about half the resident slots. Baselines are StreamingLLM, H₂O and TOVA, not Quest. | **H1:** Fig. 4a shows importance drifts, but only anecdotally, with no aggregate reuse statistic. **H2:** it is a recallable host-backed page cache, the same structure as ours. | Victims are chosen by the **current** step's score only; the kernel has no recency, frequency or history state. Selection and eviction share one score. Eviction is never ablated (no LRU vs. score) and no real-run miss rate is reported. |
| **HiSparse** (Xie et al., arXiv:2608.07009, 2026) | Exact sparse-attention decoding with full KV in host memory and a fixed B-slot GPU cache per request and layer. LRU tracks Bélády and is a good online proxy. | Per request-layer LRU. Within a step, hits are promoted above newly fetched misses (§3.2). A fused kernel marks slots hit or evictable, scans for victims and fetches misses (§3.4). | §4.3 / Fig. 6, on a GLM-5.1 DSA trace with k=2048. Mean miss rate over the first 1,000 steps: B=k 30%; at B=4096, LRU 13.4%, FIFO 17.2%, random 16.1%, Bélády 8.2%; LRU at B=8192 6.7%. §4.4 / Fig. 7: miss-resolution kernel cost against B. | **Token / KV-record** granularity: each slot holds the record for "one logical position" (§3.2). The replacement study (Fig. 6) uses one 100,384-token LongBenchV2 request. Qwen3-30B-A3B+Quest appears only in throughput experiments (§4.1–4.2). | **H2:** nearly identical setup (fixed selection, host KV, B-slot cache, FIFO/random/LRU/Bélády replay). Its hit-over-fill rule is already a binary "selected again" signal. | Selection score and rank are never used for replacement; frequency is not studied. The gap is left explicit: the Bélády result bounds "what a smarter, predictive replacement policy could still recover" (§4.3). The future-work note in §4.4 is about dynamic cache *sizing*, not replacement. |
| **Quest** (Tang et al., ICML 2024) | Query-aware page selection approximates full attention while reading a fraction of the KV. | Per-page channel min/max keys. The upper bound Σᵢ max(qᵢmᵢ, qᵢMᵢ) ranks pages and the top pages are attended (§3.4, Alg. 1). 16-token pages. | §3 and Alg. 1; page-size discussion in §4. | All KV stays resident: it "retains all of the KV cache". Host offloading or caching is not studied. | **Neither H1 nor H2.** It is the fixed selector we reuse (our pilot implements this bound on Qwen3-0.6B). | No analysis of how selected pages are reused across decode steps, and no memory hierarchy. Its ranking is never used for residency. |
| **H₂O** (Zhang et al., NeurIPS 2023) | A few heavy-hitter tokens dominate attention. Keeping heavy hitters plus recent tokens preserves quality at a small budget. | Accumulated attention score plus a recent window. Token-level and **permanent** eviction: "the evicted KV will not be seen in the future" (§5). | Attention scores follow a power law (§3, Fig. 2). Throughput and quality tables in §5. | OPT and LLaMA-family models. A lossy cache: eviction changes what is attended. | **H2 (form):** an importance-plus-recency retention score, which is the shape of our s_i. | Importance is cumulative attention, not the last top-K rank. Nothing is recallable and there is no Bélády or miss-rate comparison. Stability of heavy hitters across steps was *unverified* (that part of the paper was not read). |
| **PARROT** (Liu et al., ICML 2020) | Imitating Bélády yields a learned replacement policy that beats hand-designed ones. | LSTM over the access history, attention over resident lines, DAgger and a ranking loss, with reuse-distance prediction as an auxiliary task. | §5 and Table 1, SPEC2006 LLC: about 16% higher hit rate than LRU on average. Defines a normalized hit rate (LRU = 0, Bélády = 1). | CPU last-level cache traces. Model latency far exceeds a per-step GPU kernel budget. | **H1 (general form):** predicting reuse distance from history. **H2:** methodological precedent for normalizing against LRU and Bélády, as our G(π) does. | No model-native signal like attention rank. Its gains do not transfer to KV caching. |

## Gate A conclusion

**ArkVale does not use historical rank for eviction.**
- §5.1 says that "prior to attention computation" every page is scored "using the query and page digests". Missing top-k pages are then recalled and "bottom-ranked pages in GPU will be evicted for exchange".
- The kernel confirms this. In `select.cu`, `new_in[i] = in[pos_ids[i]]` (line 127) copies the resident slots' scores from the **current** step. Line 131 then takes the `n_recalls` lowest of them as victims. No recency, frequency or past-rank state exists.
- ArkVale's §2 presents history-based eviction as a weakness of other methods: those methods "rely on historical data to dictate cache eviction".

**No paper we read uses the last observed top-K rank as a residency signal with selection held fixed.**
- HiSparse is the closest work. It has the same cache model and baselines, uses pure LRU, and explicitly leaves the LRU–Bélády gap open.

**Gate A passes, with a narrower framing than "rank-aware eviction is new".** The open question we can claim is incremental: given fixed Quest selection and a recallable page cache, does the *last observed* selection rank carry reuse information beyond recency and frequency (H1)? And does adding it to a recency-plus-frequency policy close part of the LRU–Bélády gap (H2)?

## What we can claim

- We hold sparse selection fixed and study only residency, so every selected page stays retrievable and attention is unchanged. ArkVale ties eviction to the current ranking; we decouple the two.
- To our knowledge, based on ArkVale, HiSparse, Quest, H₂O and PARROT, none tests whether historical top-K selection rank predicts future selection beyond recency and frequency.
- HiSparse reports a 5.2-point LRU–Bélády gap at B=2k on a token-level GLM-5.1 trace and calls it the room for a predictive policy (§4.3). We measure the corresponding gap for page-level Quest selection on Qwen3-0.6B.
- We use PARROT-style normalization against LRU and Bélády for G(π).

## Claims we must avoid

- That we are the first to use attention importance for KV eviction. H₂O, ArkVale and TOVA already do; Scissorhands is cited by ArkVale but *unverified* by us.
- That we are the first host-backed or recallable KV cache. ArkVale and HiSparse are; InfLLM is *unverified*.
- That we are the first to compare LRU, FIFO and Bélády on sparse-attention traces. HiSparse Fig. 6 already does, and our FIFO, LRU and Bélády rows replicate its methodology at page granularity.
- Any direct comparison of our miss rates with HiSparse's 13.4% and 8.2%. Theirs are token-level, GLM-5.1 DSA, k=2048, one LongBenchV2 request; ours are 16-token pages, Qwen3-0.6B Quest, K=64.
- That HiSparse's Fig. 6 covers Quest or Qwen3. It does not; Qwen3+Quest appears only in throughput results. HiSparse lists Quest as a "page-level" selector (§2.1, Table 1) but manages one KV record per logical position (§3.2). Whether its Quest cache is managed per page is *unverified*.
- That ArkVale measured how often evicted pages return. Fig. 4a shows a single anecdote and Fig. 4b is an offline recall-count simulation.
- That PARROT's gains, or any learned-policy gains, carry over to KV caches.

## Consequences for the plan

1. **A current-score baseline.** ArkVale's eviction rule ("evict the resident page with the lowest current-step score") is a natural extra baseline between LRU and our policy. It needs the current scores of unselected resident pages, which trace v1 does not record. See the team decisions below.
2. **The H1 framing.** H1 should test rank against recency *and* frequency, as the blueprint says. HiSparse's hit-over-fill promotion already captures "selected repeatedly", so rank must add information beyond that.
3. **Headroom first.** If the page-level Qwen3 LRU–Bélády gap at B=128 turns out much smaller than HiSparse's token-level gap, report that as a finding before tuning δ and β.

## Unverified items

- Scissorhands and InfLLM were not read.
- H₂O: whether heavy hitters stay stable across decode steps; the later appendix was not read.
- ArkVale: OpenReview reviews and differences between submission and camera-ready versions. Only the `SelectTopkKernel` path and the inference configuration of the repository were read.
- HiSparse: the exact tie-breaking among victims, and whether its Quest deployment caches pages or tokens.
- Quest, H₂O and PARROT quotes came from the arXiv HTML and were spot-checked against PDF text. Other details of those rows were not checked line by line.

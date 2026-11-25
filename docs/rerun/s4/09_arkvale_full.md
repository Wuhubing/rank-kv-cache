# S4.9 ArkVale, full version (follow-up to S4.4), dev only

**Conclusion.** The full ArkVale is further behind the score EMA than the simplified version was, not closer.
With the official code's defaults on the Quest traces (sink and recent pages forced every step, pages leaving
the window dropped, layers 0–1 unbudgeted, cuboid-mean digest for eviction), G at B=96 is **−0.219**
[−0.236, −0.200], worse than LRU on all 14 documents. The EMA reaches 0.278 (paired ΔG −0.497 [−0.513, −0.481]);
the S4.4 simplified ArkVale had 0.214. Taking the three parts one at a time:
1. **Forced pages** barely change the access stream (+0.2 % accesses with the page-count conversion, +1.8 % with
   the token-count one), because the recent pages are almost always in Quest's top-64 anyway. LRU, EMA and Bélády
   G stay the same to the third decimal. The cost is in ArkVale's window rule: dropping the page that leaves the
   window costs ArkVale 0.04 G at B=96 and 0.06 at B=128 (0.200 vs 0.241 with Quest scores).
2. **Unbudgeted layers 0–1.** (a) Counting only layers 2–27 at the same B moves the G of LRU-relative policies (EMA,
   percentile EMA, power law, simplified ArkVale) by at most 0.003. ArkVale full itself improves from −0.245 to −0.219
   at B=96, because the cuboid-mean digest is at its worst in layers 0–1. (b) At equal total slots this is infeasible at B ≤ 128: two full layers (1 101–1 938 pages
   each) would need more than 28·B slots. It becomes feasible for 13/14 documents from B=192, and there it multiplies
   total misses for every policy: ×3.2 at B=192, ×1.7 at 256, ×1.15 at 384, neutral at 512.
3. **Digest.** Under Quest selection, evicting by any ArkVale digest is worse than evicting by the selector's own
   score. Cuboid-mean, the code's digest, scores G −0.201 at B=96 (all layers, no forcing) against +0.240 for the
   current Quest score; cuboid-center −0.110; spheres −0.42 to −1.11. LODO picks the Quest score (cuboid-max) on
   every fold. The digests correlate with the Quest score (Spearman 0.87–0.97) but disagree with it on about 20 %
   of the top-64 (cuboid-mean overlap 0.794), so eviction drops pages that Quest is about to select. With ArkVale's
   own selector end to end (exploratory, separate traces), eviction and selection use the same digest. There ArkVale
   reaches G 0.195 at B=96 (0.234 without the window rule) and the EMA on that selector's scores 0.266, a paired gap
   of −0.071 [−0.075, −0.066] that holds on 14/14 documents. **The EMA's advantage carries over to the other selector**:
   +0.031 over its current score, against +0.038 under Quest.

ArkVale with the best digest (the Quest score, everything else as released) is G 0.200 at B=96, still −0.078
[−0.081, −0.075] behind the EMA (0/14 documents better). The power-law kernel (0.310) and the all-page percentile
EMA (0.271) keep their S4 positions. Dev only, exploratory; 0 audit mismatches.

Data: 0.6B dev traces re-collected with `--record-arkvale-digest` (`data/dev_digest` on wulab2). They are
**bit-identical** to the S3 traces on all 14 documents (ranked, scores and generated tokens:
`results_v2/dev/arkvale/compare_readout_main.txt`), and the same holds on the 14 8B dev documents
(`compare_readout_8b.txt`). ArkVale-selector traces: `data/dev_arkvale_sel` (`--selector arkvale-mean`, 1 sink +
1 recent + top-62 by cuboid-mean). Replays: `analysis_v2/arkvale_full.py replay/summarize/tables`, results in
`results_v2/dev/arkvale/{main,e2e,8b}/` (`summary.csv`, `contrasts.csv`, `tables.md` with every stream, scope and
budget, `misses.csv.gz`). Statistics as in S4: G with LRU/Bélády of the **same access stream and the same layers**
as anchors, 14-document paired bootstrap (2000, seed 0), EMA / all-page-percentile EMA / power-law parameters by
LODO, ArkVale untuned (code defaults; the best-digest row is the separately listed tuned version). Every LRU and
Bélády count was recomputed with `replay.py` from forced rows built independently in the audit: main 33 536 streams,
end-to-end 6 676, 0 mismatches (`*/audit.txt`).

Names: stream `plain` = no forced pages; `page` = 1 sink + 1 recent completed page (the code's `n_sink_pages=1`,
`n_win_pages=2` at the same number of pages); `token` = 2 + 2 (same number of tokens); `page_is` / `token_is` =
InferState's default `n_sink_pages=2` (2+1, 4+2). Scope `all` = all 28 layers at budget B; `a` = layers 2–27 at B
(layers 0–1 unbudgeted, not counted); `b` = layers 2–27 at the equal-total-slots budget. `ark:<digest>:<1|0>` =
`simulate_arkvale` with that eviction estimate and the window rule on/off; **ArkVale full = `ark:cuboid_mean:1` on
stream `page`, scope `a`**.

## Final comparison: stream `page`, scope `a` (ArkVale full conditions); G [95 % CI] (miss rate %)

| Policy | B=72 | B=80 | B=96 | B=128 | B=256 |
|---|---|---|---|---|---|
| LRU | 0 (12.04) | 0 (9.37) | 0 (6.33) | 0 (3.76) | 0 (1.42) |
| Score EMA z (LODO) | 0.258 (11.33) | 0.275 (8.59) | **0.278** [0.274, 0.283] (5.70) | **0.265** [0.255, 0.274] (3.37) | 0.235 (1.32) |
| All-page percentile EMA (LODO) | 0.254 (11.34) | 0.269 (8.61) | 0.271 [0.266, 0.275] (5.72) | 0.257 [0.247, 0.267] (3.38) | 0.221 (1.33) |
| Power-law kernel (LODO) | 0.285 (11.25) | 0.293 (8.54) | 0.310 [0.305, 0.315] (5.63) | 0.293 [0.287, 0.300] (3.33) | 0.243 (1.32) |
| ArkVale simplified (S4.4: 2+2 pins, Quest score) | 0.148 (11.63) | 0.201 (8.80) | 0.214 [0.210, 0.220] (5.84) | 0.199 [0.185, 0.213] (3.47) | 0.112 (1.37) |
| **ArkVale full** (code defaults, cuboid-mean) | −0.243 (12.71) | −0.279 (10.16) | **−0.219** [−0.236, −0.200] (6.83) | **−0.083** [−0.100, −0.063] (3.88) | −0.086 (1.46) |
| ArkVale full, best digest (tuned: LODO picks the Quest score every time) | 0.195 (11.50) | 0.216 (8.76) | 0.200 [0.194, 0.206] (5.88) | 0.151 [0.134, 0.167] (3.54) | −0.023 (1.43) |
| Bélády | 1 (9.28) | 1 (6.54) | 1 (4.06) | 1 (2.29) | 1 (1.00) |

Paired ΔG (bootstrap CI; documents where the first is better / worse):

| Contrast | B=72 | B=80 | B=96 | B=128 | B=256 |
|---|---|---|---|---|---|
| ArkVale full − EMA | −0.501 [−0.511, −0.491] 0/14 | −0.554 [−0.567, −0.539] 0/14 | −0.497 [−0.513, −0.481] 0/14 | −0.348 [−0.363, −0.332] 0/14 | −0.321 [−0.350, −0.296] 0/14 |
| ArkVale full (best digest) − EMA | −0.063 [−0.065, −0.061] 0/14 | −0.059 [−0.062, −0.057] 0/14 | −0.078 [−0.081, −0.075] 0/14 | −0.114 [−0.122, −0.105] 0/14 | −0.258 [−0.292, −0.228] 0/14 |
| ArkVale full − ArkVale simplified | −0.391 0/14 | −0.480 0/14 | −0.434 [−0.450, −0.416] 0/14 | −0.282 [−0.298, −0.266] 0/14 | −0.198 0/14 |
| ArkVale full − power law | −0.528 0/14 | −0.572 0/14 | −0.529 [−0.543, −0.514] 0/14 | −0.376 [−0.391, −0.361] 0/14 | −0.329 0/14 |
| ArkVale full − all-page percentile EMA | −0.497 0/14 | −0.548 0/14 | −0.490 [−0.506, −0.473] 0/14 | −0.340 [−0.356, −0.323] 0/14 | −0.308 0/14 |

Token-count conversion (stream `token`, scope `a`) gives the same picture: ArkVale full −0.222 at B=96 and −0.084 at
B=128; ΔG vs EMA −0.501 [−0.516, −0.484] and −0.348 [−0.363, −0.331]; best digest 0.203 / 0.154. With InferState's
`n_sink_pages=2` (`page_is`, `token_is`) ArkVale full is −0.223 / −0.235 at B=96 and EMA 0.278 / 0.278.

## Item 1: forced pages (all layers, B=96 / 128)

| Stream | accesses | compulsory | LRU miss % @96 | EMA G @96 / @128 | Simplified (2+2 pins) | ArkVale, Quest score, window rule on | … window rule off |
|---|---|---|---|---|---|---|---|
| plain | 19 308 800 | 157 251 | 6.04 | 0.279 / 0.267 | 0.215 / 0.202 | 0.240 / 0.216 (= current score) | same |
| page (1+1) | 19 346 840 | 157 299 | 6.03 | 0.279 / 0.267 | 0.214 / 0.202 | 0.197 / 0.151 | 0.241 / 0.216 |
| token (2+2) | 19 651 806 | 157 490 | 6.05 | 0.279 / 0.266 | 0.242 / 0.217 | 0.201 / 0.153 | 0.242 / 0.217 |
| page_is (2+1), scope a | 19 635 391 | 157 472 | 6.06 | 0.278 / 0.264 | 0.242 / 0.214 | 0.201 / 0.152 | – |
| token_is (4+2), scope a | 20 224 103 | 157 857 | 6.10 | 0.278 / 0.265 | 0.243 / 0.216 | 0.204 / 0.156 | – |

The S4.4 penalty of the pins came from *protecting* sink and recent pages that Quest does not select: they kept
slots that the score would have freed (0.215 vs 0.240). Once the pins are part of every step's access set they
are simply accessed pages. ArkVale's eviction then equals the current-score policy when the window rule is off
(test: `test_without_window_exit_equals_score_policy_on_forced_stream`). With the window rule on, it loses
0.033–0.040 G at B=72–96, 0.06 at B=128 and 0.14 at B=256 (`ark:…:1 − ark:…:0`, 0/14 documents better): a page that
leaves the window is often selected again a few steps later and has to be recalled.

## Item 2: layers 0 and 1 unbudgeted

(a) Layers 2–27 at the same B (page stream): EMA 0.258 / 0.275 / 0.278 / 0.265 / 0.235 at B=72/80/96/128/256,
against 0.256 / 0.273 / 0.279 / 0.267 / 0.236 over all layers. Layers 0–1 are not where the EMA–LRU difference
lives, and leaving them out changes no ranking. ArkVale full (scope a) is in the final table above.

(b) Equal total slots: the limited layers get `floor((28·B − 2·P_doc)/26)` pages. P_doc ranges from 1 101 to 1 938
completed pages, so this is below K + 2 = 66 for every document at B ≤ 128 and for 1 of 14 at B=192. Per-document
budgets are in `main/eq_budgets.csv`. At B=192, 256, 384 and 512 the limited layers get 67–122, 126–191, 264–328
and 402–466 pages. G within (b), page stream:

| Policy | G (b) @192 (13 docs) | @256 | @384 | @512 |
|---|---|---|---|---|
| LRU miss % (layers 2–27) | 6.57 | 2.56 | 1.19 | 0.91 |
| Current Quest score | 0.230 | 0.180 | 0.099 | 0.094 |
| Score EMA z (LODO) | 0.268 | 0.246 | 0.236 | 0.263 |
| All-page percentile EMA (LODO) | 0.262 | 0.236 | 0.227 | 0.264 |
| Power law (LODO) | 0.280 | 0.270 | 0.239 | 0.261 |
| ArkVale simplified | 0.189 | 0.171 | 0.097 | 0.093 |
| ArkVale full (cuboid-mean, window rule) | −0.176 | −0.070 | −0.109 | −0.186 |
| ArkVale full, best digest (LODO) | 0.182 | 0.094 | −0.063 | −0.168 |

The ranking matches scope (a). But spending the slots on two full layers costs far more misses than any policy
saves. Total misses over all 28 layers (layers 0–1 have none under (b)) as a share of all accesses, against the
plain all-layer budget B on the same documents (`main/eq_vs_std.csv`; the B=192/384/512 all-layer rows come from an
extra replay, `main/misses_extra.csv.gz`, page stream, Quest and cuboid-mean digests only):

| Policy | B=192: (b) / all layers | B=256 | B=384 | B=512 |
|---|---|---|---|---|
| LRU | 6.10 % / 1.93 % (×3.16) | 2.37 / 1.36 (×1.75) | 1.11 / 0.95 (×1.16) | 0.85 / 0.84 (×1.01) |
| Score EMA z (LODO) | 5.61 / 1.76 (×3.18) | 2.15 / 1.27 (×1.70) | 1.05 / 0.92 (×1.14) | 0.83 / 0.83 (×0.99) |
| Power law (LODO) | 5.59 / 1.75 (×3.20) | 2.13 / 1.26 (×1.69) | 1.04 / 0.92 (×1.13) | 0.83 / 0.83 (×1.00) |
| ArkVale full (cuboid-mean) | 6.42 / 1.97 (×3.26) | 2.43 / 1.39 (×1.75) | 1.13 / 0.97 (×1.17) | 0.86 / 0.85 (×1.01) |
| Bélády | 4.27 / 1.23 (×3.46) | 1.47 / 0.97 (×1.53) | 0.85 / 0.83 (×1.02) | 0.78 / 0.81 (×0.96) |

At equal memory, ArkVale's unbudgeted first layers are a net loss for every policy until B ≈ 512 (B/K = 8), where
they are neutral. ArkVale's own setting (LongChat-7B, 32 layers, 4 096-token budget, contexts mostly below 32K) has
much more memory per layer relative to the context than our B=96–128.

## Item 3: digests

(i) Quest selection, eviction by each digest (`ark:<digest>:0` on stream `plain`, all layers = digest effect alone):

| Eviction estimate | top-64 overlap with Quest | Spearman with Quest | G @72 | G @96 | G @128 | G @256 |
|---|---|---|---|---|---|---|
| Quest score (cuboid-max) | 1 | 1 | 0.225 | 0.240 | 0.216 | 0.116 |
| cuboid-mean (**ArkVale code**) | 0.794 | 0.964 | −0.232 | −0.201 | −0.026 | 0.054 |
| cuboid-center | 0.815 | 0.972 | −0.185 | −0.110 | 0.035 | 0.083 |
| sphere-mean | 0.753 | 0.944 | −0.303 | −0.423 | −0.268 | −0.076 |
| sphere-center | 0.722 | 0.934 | −0.370 | −0.581 | −0.424 | −0.136 |
| sphere-max | 0.607 | 0.874 | −0.563 | −1.109 | −1.074 | −0.434 |

The order of the digests is the same as in the paper's recall-accuracy figure (cuboid better than sphere, "mean"
and "center" better than "max" for spheres). But for eviction under another selector, every digest is worse than
reusing the selector's own score, and G rises with the digest's agreement with that score. With window rule on and
forced pages (stream `page`) the order is unchanged (`main/tables.md`). LODO over the digests (window rule on) picks
the Quest score on all 70 folds in scopes all/a. In scope b it picks cuboid-center on 14 of 55 folds (B=512).

(ii) ArkVale selector end to end (exploratory; `results_v2/dev/arkvale/e2e/`, stream `page` = the pages the selector
really attends to, scope a):

| Policy | B=72 | B=80 | B=96 | B=128 | B=256 |
|---|---|---|---|---|---|
| LRU miss % | 13.11 | 10.50 | 7.46 | 4.74 | 2.03 |
| Current digest score (ArkVale eviction without window rule) | 0.233 | 0.247 | 0.235 | 0.202 | 0.126 |
| Score EMA z on the digest score (LODO) | 0.261 | 0.272 | **0.265** [0.261, 0.269] | **0.247** [0.237, 0.256] | 0.223 |
| All-page percentile EMA (LODO) | 0.258 | 0.268 | 0.261 | 0.243 | 0.220 |
| Power law (LODO) | 0.287 | 0.291 | 0.297 | 0.271 | 0.229 |
| **ArkVale (own digest, window rule)** | 0.199 | 0.215 | **0.197** [0.191, 0.203] | **0.150** [0.134, 0.163] | 0.025 |
| ΔG ArkVale − EMA | −0.061 0/14 | −0.057 0/14 | −0.068 [−0.072, −0.063] 0/14 | −0.097 [−0.104, −0.091] 0/14 | −0.197 [−0.224, −0.175] 0/14 |
| ΔG EMA − current score | +0.027 14/0 | +0.025 14/0 | +0.030 [+0.028, +0.033] 14/0 | +0.044 [+0.040, +0.048] 14/0 | +0.096 14/0 |

Generation with the ArkVale selector differs from Quest's: the outputs agree on a median of 28 tokens before the
first difference (mean 30.8; same-position agreement over the whole output 5.8 %; 10 789 vs 9 984 tokens; Quest 13 EOS + 1 loop stop,
ArkVale 14 EOS; `compare_e2e.txt`). This is ordinary sparse-attention divergence: Quest itself departs from dense
attention after a median of 14 tokens, and the ArkVale selector after 13 (`prefix_vs_dense.txt`). LRU misses more
under the ArkVale selector (7.46 vs 6.33 % at B=96, scope a), so its selections change more from step to step.

## Qwen3-8B dev (digest read-outs; streams plain/page/token, digests Quest and cuboid-mean)

QWEN8B_PLACEHOLDER

## Rule table: official code → our replay

| # | ArkVale rule (code) | Where in the code | Our implementation | Test |
|---|---|---|---|---|
| R1 | Default parameters of the released benchmarks: `page_size=32`, `page_budgets=128` pages, `page_topks=32`, `n_unlimited_layers=2`, `n_win_pages=2`, `n_sink_pages=1` | `benchmark/longbench_pred.py:33-40` (same in `performance.py:34-42`, `passkey_pred.py:137-140`); README example `README.md:39-49` | Our page is 16 tokens, K=64, B as in S4. Page-count conversion: 1 sink + 1 recent completed page (stream `page`); token-count conversion: 2 sinks + 2 recent completed pages (`token`). `InferState`'s own default `n_sink_pages=2` (`infer_state.py:31`) is reported as `page_is` (2+1) and `token_is` (4+2) | — |
| R2 | Every decode step the first `n_sink` pages and the last `n_win` pages take part in attention; `n_win` includes the page being written | `arkvale_cpp/src/select.cu:104-107` (sink and window ids copied into the output), `select.cu:99-101` (top-k runs only over pages `[ns, len+1-nw)`), `kv_cache.py:163-168` (window slots are the last `n_win` GPU slots) | The page being written is our tail, outside the cache for every policy, so `n_win=2` → 1 recent *completed* page. `with_forced` merges sink + recent pages into every step's access set; a forced page not resident is a miss | `test_forced_pages`, `test_with_forced_*` |
| R3 | A selected page that is not on the GPU is recalled into the slot of the resident, non-sink, non-window page with the **lowest current estimated score**; all victims of a step are chosen at once from the slot contents before the swap | `select.cu:122-143` (second radix select with `!select_min` over `new_in = in[pos_ids]`, `cap-ns-nw` slots) | `simulate_arkvale`: victims = n lowest `est` among residents outside the step's access set (ties: older last use, then hit/miss, then larger id, as `kvpolicy.ScoreCurrent`) | `test_without_window_exit_equals_score_policy_on_forced_stream`, `test_matches_slot_level_port_of_official_code` |
| R4 | When a new page starts and the cache is full, the page leaving the window is **dropped from the GPU whatever its score** (its slot is overwritten by the newly filled page); it can be recalled later. While the cache has room, pages are appended instead | `kv_cache.py:162-194` (`e_gci = budget - n_win + ...`, `cc2gp.scatter_(..., -1)`), `kv_cache.py:143-161` (append branch) | `simulate_arkvale(window_exit=True)` (default); `window_exit=False` reported as a variant | `test_window_exit_drops_page_leaving_window_when_full`, slot-level port |
| R5 | Top-k size: `page_topks` (default `budget // 2`) minus sinks minus (`n_win` − 1); attention uses **all** resident pages (`use_sparse_attn` is asserted off) | `infer_state.py:63-68, 82-89`; `adapter/modeling.py:151-165` (`assert not state.use_sparse_attn`, `decode_sdpa(..., kvc.c2p)`) | Selection stays Quest top-K (K=64) for (i); for the end-to-end selector (ii) K=64 = 1 sink + 1 recent + 62 by digest. Attention over the selected pages + tail, not over every resident (see Deviations) | `test_arkvale_selector` |
| R6 | The first `n_unlimited_layers` layers get no budget (all KV kept, no eviction, no recall) | `infer_state.py:57-62` (`page_budgets = [None] * n_unlimited_layers + ...`), `kv_cache.py:143` (`budget is None` → always append), `adapter/modeling.py:152` | Scope (a): G over layers 2–27 at the same B. Scope (b): equal total slots, layers 2–27 at `floor((28·B − 2·P_doc)/26)` where `P_doc` = completed pages at the end (`unlimited_budget`); infeasible where that is below K + forced pages | `test_unlimited_budget` |
| R7 | Page digest = cuboid-mean: `c = (max+min)/2` per dimension, `r = mean_k |c − k|`, bounds `c ± r`; computed once per filled page on post-RoPE keys, per KV head | `infer_state.py:448-462` (`_summarize_keys`), `:464-482` (prefill / decode), `adapter/modeling.py:124-145` (RoPE applied before) | `collect/long.py: page_digest_stats` (float32 from the bf16 keys; ArkVale uses fp16) → `digest_cuboid_mean` | `test_digest_values_match_official_code_and_paper` (literal port of `_summarize_keys`) |
| R8 | Estimated score = `sum_d max(q_d·bmax_d, q_d·bmin_d)` per query head with its KV head's digest, then **mean over query heads** (`n_groups=1`, the default) | `estimate.cu:43`, `estimate.cu:551-552` (`reshape(bsz, n_groups, H/n_groups, ...).mean(2)`), `infer_state.py:149-152` | `digest_scores(q, st, "cuboid_mean")` = `q·c + |q|·r`, mean over all query heads; the Quest score is the same formula with the raw AABB (cuboid-max) | same test, `test_digest_stats_incremental_equals_one_shot` |
| R9 | Other digests: **not in the code** (only cuboid-mean is implemented). The paper's §5.2 compares sphere-{max,center,mean} (Eqs. 1–2) and cuboid-{max,center,mean} (Eqs. 3–4); cuboid-mean is its best | paper §5.2, Fig. 7 | Recorded as `digest_cuboid_center`, `digest_sphere_{max,center,mean}`; cuboid-max = the Quest score already in the trace. ArkVale full uses the code's cuboid-mean; the best-of-variants (LODO) is listed separately | same test |
| R10 | Prefill: the last prompt token's query selects which `budget − max(2, n_sink) − n_win` pages stay on the GPU | `infer_state.py:484-516` (`ns = max(2, kvc.n_sink_pages)`), `select.cu:240-276` | Not modelled: every replayed policy starts with an empty cache (cold start), identical for all policies and anchors | — |

## Deviations (kept for every policy alike unless stated)
1. Page size 16 tokens (ArkVale 32); both conversions of sink/window counts are reported (R1).
2. The page being written is outside the cache for every policy (our S4 accounting), so ArkVale's window of 2 pages is
   1 completed page + the tail. A page that has just been completed enters the cache as a forced page and counts as
   one (compulsory) miss, although in ArkVale it is already on the GPU; these misses are the same for every policy on
   the same stream, so G is unaffected, but the miss rate is slightly higher.
3. Attention set: ArkVale attends to every resident page (R5); we keep "selected pages + tail" so that the access
   stream does not depend on the policy (otherwise trace replay is impossible). Selection (i) is Quest top-64; (ii)
   uses ArkVale's digest top-62 + the 2 forced pages.
4. Cold start instead of ArkVale's prefill eviction (R10).
5. Digest computed in float32 from bf16 keys (ArkVale: fp16 keys and arithmetic); stored as float16 like the scores.
6. Our K/B ratio (64/96, 64/128) is the S4 grid, not ArkVale's 32/128 pages.

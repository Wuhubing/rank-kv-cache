# S4.9 ArkVale, full version (follow-up to S4.4), dev only

_Status: A0 (rule table) written; results below are filled in by A1–A5._

Official code: `https://github.com/pku-liang/ArkVale`, commit `0c3b45e` (read-only clone on mini at
`~/work/rkv-arkvale/ArkVale/`). Paper: ArkVale, NeurIPS 2024 (`media/arkvale-nips24-paper.pdf` in that repository).
Implementation here: `kvpolicy_v2.forced_pages / with_forced / simulate_arkvale / unlimited_budget`,
`collect/long.py --record-arkvale-digest` and `--selector arkvale-mean`, driver `analysis_v2/arkvale_full.py`.
Tests: `tests/test_arkvale_full.py`, `tests/test_collect_long.py` (digest part).

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

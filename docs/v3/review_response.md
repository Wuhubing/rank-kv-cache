# Response to Xiaoyang Wu's review of PR #11 (R11-1 … R11-7)

回复 Xiaoyang Wu 的复核（"PR #11 复核结果与修改任务", 2026-11-27; reviewed head `f58f6d8`). Every item was first reproduced on
our side (wulab2, from the committed raw files); where our numbers differ from the review, the difference is stated and we did
not copy the review's value. All changes are on `mechanism-v3` (this PR); the R11-5 annotation of the replication-v2 tables is on
`replication-v2` (`944a9ac`) and merged here (`0fc99d5`).

| # | What we saw when reproducing (agrees with the review?) | What we did (commit) | Acceptance | Not done / differs (why) |
|---|---|---|---|---|
| R11-1 | Yes. k128 batch 32: LRU 5879 (total/256) vs 5232 (timed mean), nofetch 1924 vs 1170, allmiss 118272 vs 114688. | `e_summarize.py`: all four places use the timed steps only (`mean(step_misses[warm:])` per run, then mean over rounds); test `test_e_summarize_timed_steps_only` (`9114ef7`); all `results_v3/e/*/summary/*.csv` regenerated (`88f2c39`); docs (`cf7e269`, `abd8c86`). | LRU transfer shares equal the review's table exactly: 0.6B batch 32 / 16 = 11.4 / 7.1 %; 8B 16K b16 / 12K b32 = 5.2 / 6.2 %; page 32 / 64 / K 128 = 9.7 / 5.7 / 9.3 %; slow tier 20.9 / 35.2 / 52.2 / 68.7 %. | **Per-page cost and upper bounds differ from the review's table.** We get 2.50 µs (not 2.69) at 0.6B batch 32, bounds score / EMA / alloc+EMA / Bélády 0.80 / 0.92 / 1.37 / 3.28 ms (1.0 / 1.2 / 1.8 / 4.3 %), 8B 16K b16 EMA / alloc+EMA / Bélády 0.41 / 0.83 / 1.51 ms, 8B 12K b32 0.81 / 1.53 / 2.94 ms. The review's numbers are reproduced exactly by (all-miss − no-fetch step) / (all-miss misses − no-fetch misses), i.e. they also subtract the misses the no-fetch run *counts*. But no-fetch skips the gather and the write-back entirely (`gpu_bench_batch2.py`: `if T.policy != "nofetch": _gather_b…`), so it transfers zero pages and the step difference corresponds to all K·R·L = 57 344 all-miss pages, as `e_systems.md` describes ("/ (K·R·L)"). The independent D2 slope (b = 2.48 µs/miss at batch 32) agrees with 2.50, not 2.69. Consequently Bélády reaches ≈ 97 % of its bound (3.17 of 3.28 ms), not 92 %; we wrote "≈ 97 %" instead of "equals". Break-even recomputed from the new shares: ≥ 1 % from ~19 % share (was ~20 %), ≥ 5 % from ~52 % (was ~55 %). |
| R11-2 | Yes: `batch_8b` batch 1 35/35, batch 4 35/35, batch 8 2/35 runs with other PIDs; the others clean. | `e_systems.md` line 15 (review's wording) and the batch-4 attribution (`cf7e269`); `e_summarize.py`: `other_procs_seen` = runs with another process, new `clean` column, `max_load` = max(before, after) (`9114ef7`). | `throughput.csv`: exactly the 16 affected rows have `clean=False`; no unconditional "no other process" sentence left in `docs/v3/`. | – |
| R11-3 | Yes on the old CSV (EMA −0.92 / −0.76, alloc+EMA −0.98 / −0.80, Bélády +1.06 / +2.46 points). After R11-1 the D2 predictions change: EMA −0.95 / −0.87 / −0.64 / −0.45, alloc+EMA −1.03 / −0.96 / −0.66 / −0.41, Bélády −0.22 / −0.23 / −0.25 / −0.17 points (d = 0.04 / 0.12 / 0.28 / 0.6). | `d2_step_model.md`, `e_systems.md` (twice), `thesis_v3.md` P5: "within ~1 point for the score policies (measured 0.4–1.0 below the prediction, largest at 21–35 % share), within 0.25 points for Bélády" (`cf7e269`). | `grep "0.5 points\|0.5 percentage" docs/v3/` → no hits. | The suggested sentence was not used verbatim: with the fixed CSV Bélády is *below* the prediction by ≤ 0.25 points, not above by up to 2.5. |
| R11-4 | Yes: the text alone does not fix the implementation. Rules 1, 3, 4, 5 match `pool_run` as written; rule 2 divides by the number of pages the layer selects at that step (`len(row)`), which equals K only once the layer has ≥ K pages. | "Exact pool semantics" in `c4_allocation.md`; dated appendix appended at the end of `docs/preregistration_v3.md` ("supplementary clarification, does not change the frozen configuration or code"; byte prefix unchanged, MD5 of the first 6 385 bytes = the original file); `tests/test_pool_spec.py`: an implementation written from the appendix text alone (no project helpers) (`ffc37cf`). | Equal to `pool_run` miss for miss in all 126 combinations (3 synthetic traces incl. layers with fewer than K pages and tied scores × LRU, Bélády, EMA {z, allrank} × β {0, 0.5, 2} × α {0.3, 0.5} × 3 budgets). | Rule 2 written as "/ number of selected pages (= K once the layer has ≥ K pages)". |
| R11-5 | Yes: nofetch `ranks_equal_prep` / `tokens_equal_prep` = None in every raw file; linear DAgger is LODO (checked `efficient_dev.py` and `analysis_v3/learn.py` ourselves); memory values equal the review's table. | `e_summarize.py` writes `n/a` for unchecked; LODO label in `docs/rerun/s4/01_existing_baselines.md` and `s3_dev_vs_l4.md` (3 rows: B=96/128/256) on `replication-v2` (`944a9ac`, merged `0fc99d5`); exploratory note in `c2_learned_linear.md`; memory table in `e_systems.md` (from new `peak_decode_gib`, `host_tier_gib` columns). | nofetch rows show `n/a`; memory table: 14.17 / 16.41 / 26.18 / 19.86 / 23.76 / 29.88 GiB decode peak, 56.93 / 56.98 / 57.09 / 56.93 / 36.60 / 55.20 GiB host tier. | – |
| R11-6 | – (GPU). | `e_bench --wait-clean` (wait for a card without other processes before every run; an unclean run is repeated once and kept as `discarded_runs`) (`79c0676`); re-run in `results_v3/e/batch_8b_rerun/` (`3774b0a`), section in `e_systems.md`. | All 70 kept runs have empty before/after `other_pids` (one batch-4 LRU run was repeated); batch-4 non-LRU ratios 0.982–1.015 (no longer 0.86–0.87). Clean LRU step 44.8 ms (batch 1) and 48.0 ms (batch 4) vs 59.1 / 69.7 ms in the interfered runs. | Ran on card 3 (GPU-850490f2-…), not card 0: card 0 was taken by another user's job when the re-run started (first attempt on card 0 stopped, nothing written). |
| R11-7 | Yes for the throughput side: 8B batch 16 EMA median step 78.31 vs LRU 77.94 ms; 12K batch 32 EMA −0.2 %. After R11-1 the EMA's bound at 8B batch 16 is 0.41 ms (the review's 0.42 uses the per-page cost discussed under R11-1). | `analysis_v3/r11_netgain.py` + test (`176bf4f`); `e_bench` records memory by phase and `gpu_bench_batch2.run` separates the prefill-forward and migration peaks (`79c0676`); timing runs scripted in `runs/run_r11.sh` (`259aec6`); net-gain account from the existing runs (`5a1f9eb`) and section "Net gain" in `e_systems.md` (`44d7c8e`). | Only where an E6 timing run exists (0.6B batch 16): EMA saved transfer 0.47 − kernel overhead 0.45 = predicted 0.02 ms vs measured 0.07 [−0.07, 0.29] ms — consistent; Bélády predicted 1.55 vs measured 1.23 [0.93, 1.41] — 0.13 ms above the CI. For the three requested configurations only saved transfer and measured difference are available (table below): implied overhead 0.7–1.1 ms/step for the score policies, 0.1–0.35 ms for Bélády. | **The 8B batch 16 / 12K batch 32 and 0.6B batch 32 kernel-timing runs and the memory-by-phase measurement were not done:** from 00:56Z to 06:56Z on 2026-11-28 every usable card (0, 1, 3; card 2 excluded) had another user's process, so the 6-hour waiting limit of the task was reached. A first start on card 0 at 00:44Z was stopped because another user's training job appeared during its preparation run (nothing written). The scripts are ready (`runs/run_r11.sh`, items `timing_8b/*`, `timing_06b_b32`). |

## d2_model.csv, batch-32 rows after the fix
`results_v3/e/batch_06b/summary/d2_model.csv` and `results_v3/e/rest/summary/d2_model.csv` (a, b, c in ms; transfer share and
per-page cost recomputed with the timed-step misses; a, b, c unchanged):

| config | a | b (µs/miss) | c | eff. GB/s | LRU misses/step | LRU mean step | transfer share | all-miss | no-fetch | transfer max | µs per missed page |
|---|---|---|---|---|---|---|---|---|---|---|---|
| batch_06b | 68.63 | 2.484 | 13.89 | 26.4 | 3583.2 | 77.90 | 11.43 % | 210.72 | 67.53 | 143.19 | 2.497 |
| ps32_06b | 114.01 | 4.933 | 28.41 | 26.6 | 2491.8 | 126.84 | 9.69 % | 399.40 | 112.79 | 286.61 | 4.998 |
| ps64_06b | 211.02 | 10.172 | 57.53 | 25.8 | 1267.1 | 224.47 | 5.74 % | 782.24 | 209.55 | 572.69 | 9.987 |
| k128_06b | 121.29 | 2.389 | 16.22 | 27.4 | 5231.6 | 134.60 | 9.29 % | 405.88 | 119.94 | 285.94 | 2.493 |
| batch_8b_12k | 126.80 | 2.534 | 18.03 | 25.9 | 3308.4 | 135.60 | 6.18 % | 309.18 | 125.29 | 183.89 | 2.494 |

## R11-7 net-gain table
`results_v3/e/netgain/summary/netgain.csv` (`analysis_v3/r11_netgain.py`). Saved transfer = (LRU − policy misses per timed step)
× per-page cost (E3); measured = LRU − policy mean step over the timed steps, paired by round, bootstrap 95 % CI over 5 rounds;
implied overhead = saved − measured. Policy-kernel overhead (CUDA events, e2→e3, policy − LRU) is available only for 0.6B batch 16
(E6, 2 rounds); the planned R11-7 timing runs were not done (see the R11-7 row). All throughput runs used here are clean.

| config | policy | saved transfer (ms/step) | policy overhead vs LRU | predicted net gain | measured step difference [95 % CI] | implied overhead |
|---|---|---|---|---|---|---|
| 0.6B batch 16 | EMA | 0.47 | 0.45 [0.45, 0.45] | 0.02 | 0.07 [−0.07, 0.29] (consistent) | 0.40 |
| 0.6B batch 16 | Bélády | 1.67 | 0.12 [0.11, 0.12] | 1.55 | 1.23 [0.93, 1.41] (0.13 above) | 0.43 |
| 0.6B batch 32 | current score | 0.80 | not measured | – | 0.10 [−0.06, 0.37] | 0.71 |
| 0.6B batch 32 | EMA | 0.92 | not measured | – | 0.08 [0.05, 0.10] | 0.84 |
| 0.6B batch 32 | allocation + EMA | 1.37 | not measured | – | 0.48 [0.44, 0.52] | 0.89 |
| 0.6B batch 32 | Bélády | 3.28 | not measured | – | 3.15 [3.13, 3.17] | 0.13 |
| 8B 16K batch 16 | current score | 0.32 | not measured | – | −0.36 [−0.51, −0.09] | 0.67 |
| 8B 16K batch 16 | EMA | 0.41 | not measured | – | −0.42 [−0.51, −0.34] | 0.83 |
| 8B 16K batch 16 | allocation + EMA | 0.83 | not measured | – | −0.00 [−0.29, 0.25] | 0.83 |
| 8B 16K batch 16 | Bélády | 1.51 | not measured | – | 1.16 [1.09, 1.21] | 0.35 |
| 8B 12K batch 32 | current score | 0.64 | not measured | – | −0.48 [−0.56, −0.39] | 1.12 |
| 8B 12K batch 32 | EMA | 0.81 | not measured | – | −0.25 [−0.29, −0.21] | 1.05 |
| 8B 12K batch 32 | allocation + EMA | 1.53 | not measured | – | 0.48 [0.43, 0.52] | 1.06 |
| 8B 12K batch 32 | Bélády | 2.94 | not measured | – | 2.69 [2.68, 2.70] | 0.25 |

Reading: where both terms were measured, the EMA's extra kernel time cancels its saved transfer and the account closes within the
CI. The 0.6B batch-32 EMA gap you asked about (bound 0.92 ms, measured +0.08 ms) is 0.84 ms of implied overhead; whether the kernel
alone explains it at batch 32 and on 8B (36 layers) is exactly what the missing timing runs would show. Scaling the 0.6B batch-16
per-layer EMA cost (16 µs) to 36 layers gives ≈ 0.58 ms, which would leave ≈ 0.25 ms of the 8B batch-16 gap unexplained if the
per-layer cost did not grow with the batch — an extrapolation, not a measurement.

Tests: `pytest tests validation` on wulab2 at `abd8c86`: 596 passed, 9 skipped (GPU tests, run with `CUDA_VISIBLE_DEVICES=`); the
`gpu_bench_batch2` GPU tests (7) passed after this round's change on card 0 (23:42Z) and card 3 (00:01Z).

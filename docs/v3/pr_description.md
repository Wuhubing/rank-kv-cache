# mechanism-v3 → replication-v2: mechanisms, diagnoses of the negative results, a third preregistered held-out set

Task: `docs/workshop/TASK_v3.md` (followup-v3). Part A (fixes R1–R13) is committed directly on `replication-v2` and listed in
`docs/rerun/STATUS_fixes.md`; this PR adds parts B–G. Conclusions: `docs/v3/thesis_v3.md`; one page per item in `docs/v3/`.

## Results in one paragraph
The score EMA's ~28 % of the LRU–Bélády gap replicates a third time on new documents (0.6B 0.281, 8B 0.277; preregistered).
Two new policies chosen on dev are confirmed on that set: a **normalised power-law kernel** of the same score signal (+0.048 /
+0.043 G over the EMA, 21/21 documents each) and a **global slot pool** ranked by the score percentile that needs no offline
allocation curves (+0.119 / +0.233 over uniform budgets; 96 % / 91 % of the offline allocation's gain at B=96). Mechanistically,
the policy's job is to predict which pages *the selector* will pick in the next ~16 steps: true attention is a worse signal than
the selector's own score (G −0.86), any mismatch between the two pushes eviction below random, the decisive information is the
current score of the residents just below the cut (refreshing only 16 of them recovers 91 %), and richer learned models do not
beat the power-law kernel. E (fixed batched path, 5 randomised rounds): on a PCIe 4 GPU transfer is ≤ 11 % of the decode step, so the EMA gains ≤ 1 % throughput and Bélády 1.5–4 % (the score policies' own kernel time takes most of their saved transfer); with a simulated slower tier the gains grow in proportion to the miss reduction (up to +7 % / +34 %).

External evidence integrated with attribution (`docs/v3/external_xiaoyang.md`): Xiaoyang Wu's M1 Max replication and new-task
results (PR #9), his L4 batched transfer benchmark (PR #9; transfer share ~14 % caps speedups at a few percent) and his
rank-decision analysis (PR #10). Not re-run; key numbers re-computed from his committed files (`analysis_v3/xy_check.py`); his
data files are referenced, not copied.

## Candidate conclusions P1–P6 (details: `thesis_v3.md`)
| | Verdict | Confirmatory? |
|---|---|---|
| P1 mechanism law (G predictable from trace statistics) | weakened: Spearman 0.60 / 0.77 (B/K 1.5 / 2), synthetic model reproduces the α-curve shape only | – (dev) |
| P2 better simple policy | supported: normalised power law, ΔG +0.048 / +0.043 | **yes (H2)** |
| P3 free memory saving | supported: 8–10 % more slots at B=96, 16–18 % at B=128 (allocation + EMA); held-out post-hoc consistent | dev + post hoc |
| P4 allocation without offline curves | supported against uniform budgets; ≥ 80 % of the offline gain only at B=96 | **yes (H3)** |
| P5 boundary of system gains | supported: upper bound holds within noise (Bélády ≈ 97 % of it), speedup ∝ transfer share × miss reduction (D2 within ~1 point); real-hardware gains ≤ 1 % (EMA), 1.5–4 % (Bélády); score policies lose 0.7–1.1 ms/step of their bound to their own cost; larger gains only in simulation | – |
| P6 explanations of negative results | supported (staleness, classic-policy defaults, pins/protection and the accumulated-score flip) | – (dev) |

## What was done
- **B** (dev, 0.6B and 8B): kernel factorial and explicit residency terms (B1); trace statistics, LODO regression, synthetic model
  (B2); finite-lookahead Bélády, error classes, boosted-tree upper bound (B3); a new trace read-out of the dense attention of every
  page (`collect.long --record-attn-all`, bit-identical selection) and replays with it (B4); phases (B5); spatial structure (B6);
  update-scope and reference-set curves, staleness read-out (B7).
- **C**: the frozen simple policy (C1), a 13-feature linear model with DAgger (C2), the initial value (C3), online ghost
  reallocation and a global pool (C4), cost curve (C5).
- **D**: capacity (D1), noise models incl. time-correlated, order-preserving, measured, counterfactual (D3), classic policies (D4),
  pins/protection incl. the CO2 flip (D5), α (D6), high budgets (D7); D2 is the step-time model of part E.
- **E**: a fixed batched path `gpu_bench_batch2.py` (pinned host tier, fused batched policy kernel without host sync, zero-copy
  fetch, async write-back; CUDA tests) and a randomised, multi-round driver; E2 batch 1–32, E3 all-miss/no-fetch bound, E4 tail attribution, E5 page 32/64, K 128, 8B and a simulated slow tier, E6 policy kernel time, D2 step model (`e_systems.md`, `d2_step_model.md`). Transfer/compute overlap was not achieved (see deviations).
- **F**: document pool survey (2 262 unused documents; `splits/longctx_v3.json`), preregistration (commit `087d53f`, pushed
  00:15:31Z) before collection (00:15:39Z), one evaluation; all hypotheses pass on both models (`f3_heldout.md`).
- **G**: post-hoc capacity on the held-out sets (G1), 8B decoded to EOS (G2: G 0.278, unchanged).

## New code (all with tests; `pytest tests` passes)
`kvpolicy_v3.py` (kernels, explicit-term EMAs, two-scale, lookahead, scoped / reference-rank EMA, shrunken initial value,
protection/pins incl. accumulated score, per-step simulator); `analysis_v3/` (run + experiment registry, b2_stats, b2_model,
b4_signal, instrument, inst_summary, learn, learn_summary, noise, c4_alloc, c5_cost, d1_iso, d5_summary, d6_alpha, e_bench,
e_summarize, f_eval, freeze_v3); `gpu_bench_batch2.py`; `collect/long.py --record-attn-all`; `collect/sources.py` parquet source;
`analysis_v2/survey_v3.py`. `kvpolicy.simulate` gains a `needs_future` hook (no behaviour change for existing policies).

## Checks
Every replay's LRU and Bélády re-checked with `replay.py` (`analysis_v2.audit_v2`, which now fails on missing reference rows):
0 mismatches in every audit (`results_v3/**/audit.txt`). The C4 pool's global LRU/Bélády are a different cache model and are not
audited by `replay.py` (its per-layer curves are).

## Deviations and process notes
1. The C4 dev replays (which chose the P4 configuration) ran on the mini on copies of the dev traces, before the user decided
   (2026-11-25) that all computation runs on wulab2; everything after that ran on wulab2.
2. 01:38–01:40Z the boosted-tree run used all cores (scikit-learn's OpenMP default), host load reached 50 (> the 44 limit); stopped
   and restarted with one thread per process.
3. Briefly more than 16 processes (before the limit was raised to 32): a 2-process read-out (B4 signal) and the G2 collection
   process ran next to a 16-process queue.
4. The first E runs crashed (card 2: "illegal instruction" / "illegal memory access"; card 0: a segfault in
   `torch.cuda.empty_cache`, then "illegal memory access" in Bélády runs). Replacing `Tensor.record_stream` by per-step references
   fixed the segfault but not the illegal accesses, which occurred only in long multi-run benchmarks with the fetch on a side
   stream and never under `CUDA_LAUNCH_BLOCKING=1`: an unresolved race in the overlapped path. Part E therefore runs with
   `overlap=False` (fetch and write-back on the main stream; still pinned, sync-free, fused policy kernel). The E1 requirement
   "transfer overlapped with compute" is not met. Card 2 is not used (it also faulted in plain model code during the ArkVale task).
   Failed attempts are kept on wulab2 (`out/v3/e/failed/`) and not used.
5. scikit-learn 1.6.1 was added to the project venv on wulab2 (and the mini's) for B3's trees.
6. The F3 evaluation paused the C/D queue at a job boundary to stay within 32 processes; no effect on results.
7. The R10 note written in part A first said the 8B steps were "all thinking"; G2's manifests showed thinking ends at step
   396–843, and the note was corrected on `replication-v2` (`9058540`).
8. mechanism-v3 was created before part A was finished and `replication-v2` was merged into it (not rebased) as part A progressed.
9. The task's section 7 gate for P4 was evaluated "against uniform budgets + EMA" (written in `c4_allocation.md`).
10. Review of PR #11 (2026-11-27/28, R11-4): a dated appendix was appended at the end of `docs/preregistration_v3.md` (exact pool
    semantics; clarification only, the earlier text, the frozen configuration and the code are unchanged).
11. R11-6 ran on card 3, not card 0 (card 0 was taken by another user's job; a first attempt on card 0 was stopped, nothing
    written). R11-7's timing runs were not done: no card met the preconditions for 6 h (00:56–06:56Z on 2026-11-28).
12. First pytest run of the review round on wulab2 started without `CUDA_VISIBLE_DEVICES` and without a thread limit (~40 cores
    for about a minute); stopped and re-run with `CUDA_VISIBLE_DEVICES=` and 4 threads. No GPU process was created.

## Revisions after Xiaoyang Wu's review (2026-11-27/28; `docs/v3/review_response.md`)
- R11-1: misses per step in `e_summarize.py` count the timed steps only; all E summaries regenerated. LRU transfer share at
  batch 32: 11.4 % (was 12.5 %), per missed page 2.50 µs (was 2.42), Bélády ≈ 97 % of its bound; break-even ~19 % / ~52 %.
- R11-2: the 8B batch-1/4 runs (and two batch-8 runs) had other processes on the card: `clean` column, docs corrected.
- R11-3: D2 accuracy stated as a range (score policies within ~1 point, 0.4–1.0 below; Bélády within 0.25).
- R11-4: exact global-pool semantics (`c4_allocation.md`, dated appendix of the preregistration, `tests/test_pool_spec.py`).
- R11-5: `n/a` for unchecked nofetch ranks/tokens; linear DAgger labelled LODO (replication-v2); C2 marked exploratory; GPU
  memory table.
- R11-6: clean re-run of 8B batch 1 and 4 (`results_v3/e/batch_8b_rerun/`): launch-bound, ratios within noise of 1.
- R11-7: net-gain account from existing runs (`results_v3/e/netgain/`); the 8B / 0.6B batch-32 kernel timing and memory-by-phase
  runs were not done (no clean card within 6 h).

Confirmatory: the three hypotheses of `docs/preregistration_v3.md` (and the two earlier confirmations). Everything else is
exploratory (dev) or labelled post hoc (G1).

🤖 Generated with [Claude Code](https://claude.com/claude-code)

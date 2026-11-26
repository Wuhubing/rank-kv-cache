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
beat the power-law kernel. E: E_SUMMARY_PLACEHOLDER

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
| P5 boundary of system gains | P5_PR_PLACEHOLDER | – |
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
  fetch on a side stream, async write-back; CUDA tests) and a randomised, multi-round measurement driver. E_DONE_PLACEHOLDER
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
4. The first E runs crashed: on card 2 ("illegal instruction" / "illegal memory access", three times) and then on card 0 (segfault in
   `torch.cuda.empty_cache` after several runs in one process). The cause was in `gpu_bench_batch2.py` (tensors handed to the side
   stream via `Tensor.record_stream`); they are now kept alive until the end of each step, and repeated multi-round runs pass.
   Card 2 is still not used for E because it also faulted in plain model code during the ArkVale task. The failed attempts'
   outputs are kept on wulab2 (`out/v3/e/failed/`) and not used.
5. scikit-learn 1.6.1 was added to the project venv on wulab2 (and the mini's) for B3's trees.
6. The F3 evaluation paused the C/D queue at a job boundary to stay within 32 processes; no effect on results.
7. The R10 note written in part A first said the 8B steps were "all thinking"; G2's manifests showed thinking ends at step
   396–843, and the note was corrected on `replication-v2` (`9058540`).
8. mechanism-v3 was created before part A was finished and `replication-v2` was merged into it (not rebased) as part A progressed.
9. The task's section 7 gate for P4 was evaluated "against uniform budgets + EMA" (written in `c4_allocation.md`).

Confirmatory: the three hypotheses of `docs/preregistration_v3.md` (and the two earlier confirmations). Everything else is
exploratory (dev) or labelled post hoc (G1).

🤖 Generated with [Claude Code](https://claude.com/claude-code)

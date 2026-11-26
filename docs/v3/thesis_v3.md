# Thesis after followup-v3 (mechanism-v3)

Basis: `~/work/rkv-followup/TASK.md` (docs/workshop/TASK_v3.md) sections 2–3; replication-v2 (`docs/rerun/thesis.md`, corrected
in part A, `docs/rerun/STATUS_fixes.md`). **Confirmatory** = preregistered held-out evaluated once (`docs/preregistration_v3.md`,
`docs/v3/f3_heldout.md`, plus the two earlier confirmations). **Exploratory** = dev only. Every row cites its page.

## Revised thesis
> In a sparse-attention KV cache with host offload, the replacement policy's job is to predict **which pages the selector will pick
> in the next few steps**. The selector's own scores of the pages just below the selection cut, remembered over time, are the
> signal for that.
> 1. A per-page EMA of that signal closes ~28 % of the LRU–Bélády gap on two models, three document sets (confirmatory), three
>    collection hardware types (L4, RTX 6000 Ada, and an independent replication on Apple M1 Max by Xiaoyang Wu, PR #9), and every
>    setting tried; a **power-law-weighted mean** of the same signal closes ~32–33 % (confirmatory on the third set), and a
>    **global pool** ranked by the same signal's percentile replaces offline per-layer budget curves at tight budgets
>    (confirmatory against uniform budgets).
> 2. What limits the policy is the unpredictable part of the next ~16 steps; better importance estimates (true attention) do not
>    help, and any mismatch between the policy's signal and the selector's makes it worse than random.
> 3. The savings become wall-clock gains only when transfer is a large share of the decode step: P5_PLACEHOLDER

## The six candidate positive conclusions (task section 3)
| # | Candidate | Verdict | Evidence |
|---|---|---|---|
| P1 | Mechanism law: G is set by the time correlation of the unselected pages' ordering and predictable from trace statistics | **Weakened (not supported as a law).** The support criterion (Spearman ≥ 0.7 over layers × settings at the primary budget and a synthetic model reproducing the curves) is not met: 0.60 at B/K 1.5, 0.77 at B/K 2; the synthetic model reproduces the shape of the α curves but not the level. What stands: "G is stable at 0.27–0.31" as an empirical fact, explained by the trace statistics being nearly identical across settings; and the mechanism pieces of B3/B4/B7 (next-16-step prediction, selector's own score, current scores near the cut). | `b2_stability.md`, `b3_remaining_gap.md`, `b4_true_attention.md`, `b7_global_ordering.md` |
| P2 | A better simple policy: EMA + the term separated in B1, O(1) state per page | **Supported, confirmatory.** The term is not a residency penalty (explicit penalties hurt) but the kernel's shape and normalisation: a normalised power-law kernel of z (γ=1). Dev: +0.042 [0.036, 0.048] (0.6B), +0.047 [0.043, 0.051] (8B) over the EMA at B=96. Preregistered third held-out: **+0.048 [0.044, 0.051] (0.6B) and +0.043 [0.040, 0.046] (8B), 21/21 documents each** (H2). State: 16 floats per page (O(1)). | `b1_powerlaw.md`, `c1_simple_policy.md`, `f3_heldout.md` |
| P3 | Free memory saving: policy + allocation at the same miss rate | **Supported (dev; post-hoc held-out replay consistent).** At B=96 offline allocation + EMA equals LRU with 8–10 % more slots (0.6B / 8B; 14 / 22 MiB per sequence at 32K), the global pool 8–10 % (14 / 21 MiB), C1 alone 6–7 %, the EMA alone 5–6 %; at B=128 16–18 %, 12–13 %, 8–9 %, 6–7 %. On the three evaluated held-out sets (fixed configurations, labelled post hoc) every ratio is within 0.01–0.02 of dev. | `d1_capacity.md`, `g1_posthoc.md` |
| P4 | Allocation without offline curves | **Supported with a stated limit, confirmatory against uniform budgets.** The global percentile pool beats uniform budgets + EMA on the third held-out (**+0.119 [0.114, 0.125] on 0.6B, +0.233 [0.217, 0.249] on 8B**, 21/21; H3) and reaches 96 % / 91 % of the frozen offline allocation's gain at B=96 (dev: 95 % / 93 %, the ≥ 80 % criterion met). At B=128 it reaches only 63–77 % (dev 60–74 %), so "online reaches ≥ 80 % of offline" holds at tight budgets only. The ghost-list online reallocator: 79 % at B=96. | `c4_allocation.md`, `f3_heldout.md` |
| P5 | Boundary of system gains | P5_ROW_PLACEHOLDER | `e_systems.md`, `d2_step_model.md` |
| P6 | Explanations of the negative results (classic policies, pinned pages, selected-only rules) | **Supported.** Selected-only rules: stale values freeze into an inverted ranking (Spearman of stored vs current z −0.35 vs +0.84 with full updates); refreshing just the 16 lowest-scoring residents recovers 91 %. Classic policies: frequency does not saturate (AUC 0.65) but their defaults (demote new pages, protect frequent ones) are wrong when 84 % of the selection persists and the decisive information is the current rank near the cut. Pins/protection: the sink page and the newest pages are selected 83–98 % of steps anyway, so pinning them is neutral and pinning more costs slots; protection hurts unless the score is biased against new entries (D5). | `b7_global_ordering.md`, `d4_classic_policies.md`, `d5_pins_protection.md` |

## The seven negative results (task section 1.3), diagnosed
| # | Negative result | Diagnosis | Page |
|---|---|---|---|
| N1 | Capacity matters more than the policy | Holds for the EMA alone (worth 5–7 % more slots; capacity B 96 → 128 removes 41 % of LRU's misses), but policy and capacity are not alternatives: at every budget the policy is a free capacity gain, and with allocation it is worth 8–18 % more slots at B/K 1.5–2 (peak at B/K ≈ 2), 10–56 MiB per sequence; the global pool reaches 20–25 % at B/K 4. Bélády (18–46 %) remains far ahead. | `d1_capacity.md` |
| N2 | Saved bytes do not become latency or throughput | N2_PLACEHOLDER | `e_systems.md`, `d2_step_model.md` |
| N3 | ~72 % of the gap remains | Holds, and is now located: 96 % of it is closed by knowing only the next 16 steps; 71 % of the EMA's wrong evictions return within 16 steps, 65 % sit at the 0.8–0.9 score percentile just below the cut. Richer features do not close it: a 13-feature linear model (0.313) and boosted trees with interactions (0.315–0.321 at B=96 on both models) are no better than the single power-law kernel (0.320 / 0.319). The remaining gap is the part of the next few selections that the current and past scores do not predict. | `b3_remaining_gap.md`, `c2_learned_linear.md` |
| N4 | Noisy scores push the policy below random | Holds only when the policy's scores disagree with the selector's; a rank correlation of 0.95 between them is enough to erase the gain, persistent (AR) noise is worse than independent noise. When the same noisy scores drive the selection, the policy keeps G 0.16–0.34. Quest's error against true attention (ρ ≈ 0.73–0.81) is not this kind of noise: evicting by true attention is itself worse than random (−0.86). Shrinking the initial value does not help. | `d3_noise.md`, `b4_true_attention.md`, `c3_initial_value.md` |
| N5 | α too large is worse than LRU | Holds at small budgets; the best α grows with B/K (0.4 → 0.9 (0.6B) and 0.5 → 0.95 (8B) from B/K 1.25 to 4 on dev; the synthetic model reproduces the trend and the collapse at α → 1). Mechanism: at small B the candidates return within a few steps, so the latest score matters and a long memory keeps stale values; at large B they sit deeper below the cut where averaging helps. A tuning-free per-layer α rule derived from the synthetic model fails (−0.02 to −0.08 vs LODO); per-layer LODO α adds only +0.002–0.007; α ≈ 0.5 at B/K ≤ 1.5 rising to 0.9–0.95 at B/K 4 is the practical rule. | `b2_stability.md`, `d6_alpha.md` |
| N6 | Pinned pages and protection zones hurt | Pinning the first page or the newest pages is neutral (they are selected 83–98 % of the time); pinning 2–4 sink pages costs 0.03–0.08 G (pages 1–3 are selected only 5–10 % of steps). A FIFO protection zone hurts LRU and the EMA at every budget, in every layer group, and in every synthetic regime. The flip to "protection helps" appears with an H₂O-style accumulated score, which is biased against new entries (CO2's setting): there a zone of ρ ≈ 0.2–0.3 halves the loss (0.6B −0.77 → −0.39, 8B −0.98 → −0.47). | `d5_pins_protection.md` |
| N7 | Wide CI of G at high budgets | A property of the metric: above B/K = 4 the LRU–Bélády gap is a few thousand misses, 97 % of misses are compulsory. Report absolute savings and iso-miss capacity there (EMA still worth 3–7 % more slots; repeat-miss drop grows to 25 %). | `d7_high_budget.md` |

## External evidence (Xiaoyang Wu, PR #9 / #10; `external_xiaoyang.md`)
Not re-run by us; key numbers re-computed from his committed result files. (a) M1 Max (MPS) replication of the original held-out
test: EMA G 0.280 [0.276, 0.283], 21/21, allocation + EMA ΔG +0.120 (all three preregistered criteria pass; checked). (b) New
documents / task, exploratory: NarrativeQA G 0.283 (8/8), truncated GovReport 0.284 (3/3) (checked; some documents overlap our 64K
dev setting). (c) L4 batched transfer benchmark: ~5.3 µs per page, LRU transfer share saturating at ~14 % of the step at B=96,
hence step-speedup ceilings of +5.4 % (Bélády), +2.1 % (allocation + EMA), +1.5 % (EMA) — transfer and compute timed separately
and added, not end to end (checked from his CSV). Used next to our part E to explain why the ceiling is the transfer share, not as
a replacement. (d) Adding the rank term changes 2.93 % of eviction decisions and saves 0.072 % of repeat misses on his 20 dev
documents (checked), consistent with stage 1–2's post-hoc result and B7.

## Corrections carried over from part A (replication-v2)
The power-law explanation, the Holt sentence, the 97 % baseline, the batch-32 "gain", S3-FIFO, the tests, the AUC censoring,
the post-hoc listings, the 8B scope (first 1 023 steps contain the end of thinking and 180–627 answer steps), provenance,
degenerate generations and the noise model: `docs/rerun/STATUS_fixes.md`.

## Limitations
- One selector family (Quest upper bound, page 16, K=64 except the K/page sweeps), one model family (Qwen3 0.6B / 8B), one GPU type.
- The third held-out set is GovReport only (the pool's QMSum share was 2.5 %); 8B traces are capped at 1 024 tokens (G2 shows the
  result holds for full decodes on dev).
- P1's regression and the synthetic model are descriptive; the model does not reproduce the level of G.
- The global pool was evaluated in the trace simulator only (no system implementation in part E).
- D3's counterfactual keeps the generated tokens fixed.

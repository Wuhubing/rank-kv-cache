## What this PR is

A complete re-run of the score-EMA KV page-cache study (`docs/workshop/rerun_plan.md`, S0–S8) on one RTX 6000 Ada (wulab2
card 0, UUID `GPU-a74391df-5c27-152f-d0c0-6b86f8ea07d3`). Every trace was collected again from scratch. The missing metrics and
baselines (plan §5–6) were added. The study was extended to Qwen3-8B with a new preregistered held-out set, and batch > 1 was
measured. **Not for merging.** It reports results and leaves the existing branches and frozen files untouched.

Per-step write-ups: [`docs/rerun/`](docs/rerun/): environment, S3 dev vs L4, S4 (one page per experiment), S5 replication,
S6 extensions, S7 systems, thesis, status. Small result files: `results_v2/`. Raw traces stay on wulab2 (sha256 in every
manifest); nothing was uploaded.

## Preregistrations (committed and pushed before the data they govern)
| Preregistration | Commit (UTC) | First held-out trace started (UTC) | Result |
|---|---|---|---|
| [`docs/replication_v2.md`](docs/replication_v2.md): 0.6B, original held-out, frozen L4 parameters | `86f1166` 2026-11-25T06:58:09Z | 06:58:24Z | **All 3 criteria pass**: EMA G 0.279 [0.275, 0.282] (L4 0.281), 21/21, ΔG(alloc) 0.119 [0.112, 0.127] |
| [`docs/preregistration_8b.md`](docs/preregistration_8b.md) + `splits/heldout_frozen_8b.json`: Qwen3-8B, new held-out | `328059e` 2026-11-25T08:35:29Z | 08:35:54Z | **All 4 criteria pass**: G 0.272 [0.269, 0.276], 21/21, ΔG 0.252 [0.243, 0.260], within-allocation G 0.273 [0.268, 0.277] |

Each held-out set was collected once and evaluated once with the preregistered commands.

## Thesis check (plan §1.2), point by point (full table: [`docs/rerun/thesis.md`](docs/rerun/thesis.md))
**1. "An EMA over the score history of all pages steadily narrows the LRU–Bélády gap; per-layer budgets reduce misses further."**
- ✅ **Supported, confirmatory on two models and two GPUs**: G ≈ 0.27–0.28 on every held-out set; every document of every
  setting beats LRU (0.6B, 8B, 64K/YaRN, RULER, SCBench multi-turn, page 8–64, K 32–128, sampled decoding).
- 🔁 **Mechanism reworded** (D4: "score magnitude" vs "global ordering" → **global ordering**). An EMA of each page's percentile
  rank among all pages keeps 97 % of the gain (0.271 vs 0.279); magnitude adds only 0.0075 [0.0064, 0.0088]. Updating only selected
  pages, or InfLLM's selected-only rules (even with the true attention mass), is ≈ LRU or worse.
- ✅ **Per-layer budgets**: supported against the correct anchor. Against the same-allocation Bélády, allocation + EMA closes 35 %
  (0.6B) / 43 % (8B) of the gap; within an allocation EMA still closes 28 % / 27 %, so the two add up.
- ⬇️ **Weakened / qualified**: the gain equals only 5–7 % more LRU slots (Bélády: 18–46 %), so capacity matters more than the policy.
  Noisy scores push the policy *below random*, not back to LRU. "Best policy" is not claimed: a power-law age kernel (0.310) and
  a learned linear model (0.300) beat the EMA on dev, and the EMA is one point on a curve from current score (ArkVale's eviction
  rule, α=0) to frequency (α→1).
- ✅ **ArkVale as released is weaker, not stronger** (S4.9, dev, exploratory): with the official code's defaults ArkVale reaches
  G −0.219 at B=96 (worse than LRU; EMA 0.278). Its cuboid-mean digest disagrees with the Quest selector on ~20 % of the top-64,
  so evicting by it drops pages Quest is about to select; with the selector's own score it reaches 0.200 (ΔG −0.078 vs EMA). With
  ArkVale's own selector end to end the EMA still beats ArkVale by 0.068 and gains +0.030 over the current score (14/14).
  Qwen3-8B dev agrees (ArkVale full −0.344 vs EMA 0.270 at B=96).

**2. "On common hardware these miss and byte savings do not necessarily become latency savings; we characterise when they do."**
- ✅ **Supported and now measured with batching**: EMA moves 8–11 % fewer bytes (allocation + EMA 11–20 %, Bélády 18–40 %), and
  measured misses equal the simulator. Batch-1 latency is a tie for every policy (0.6B 30.1–31.0 ms/token at B=80–256; 8B 40.6 ms).
  With batching (new `gpu_bench_batch.py`, one merged H2D transfer per layer) throughput scales almost linearly; up to batch 24
  EMA reaches 0.90–1.00× LRU (Bélády +0–7 %). At batch 32 on 0.6B the observed difference (EMA +3.4 %,
  allocation + EMA +4.1 %, Bélády +4.8 % tokens/s) comes from LRU's long-tail steps: the median step is the same (92.7–93.0 ms for
  LRU, current score, EMA), so it **cannot be attributed to the policy** (fixed run order, 2 repeats; corrected in followup-v3, R4).
  On 8B at batch 32 there is none (compute-bound step, ±2.5 % noise).
- ❌ **Removed**: the "batch 32 saves ~2.3 ms/step" linear extrapolation (README, STATUS_REPORT, gpu_benchmark.md now point to the
  measurement).
- The characterisation is stated as conditions that we did *not* measure (slower link, MHA-size pages, more pages per step, fully fused
  async offload).

## Main numbers next to L4
| | L4 (original) | RTX 6000 Ada (this PR) |
|---|---|---|
| dev, B=96: LRU miss / EMA G | 6.03 % / 0.279 | 6.04 % / 0.279 |
| 0.6B held-out | 0.281 [0.277, 0.285], 21/21 | 0.279 [0.275, 0.282], 21/21 |
| 8B new held-out | – | 0.272 [0.269, 0.276], 21/21 |
| batch-1 latency, B=96 (LRU / EMA) | 62.3 / 62.3 ms | 30.3 / 30.3 ms |
| batch > 1 | extrapolated ("batch 32 saves ~2.3 ms/step") | measured 1–32: no gain up to 24; batch 32 (0.6B): difference from LRU's long-tail steps, same median step, not attributable to the policy; 8B none |

## What was added
- `collect/long.py`: the long-trace capture ported from the notebook. Same selector, prompts, stop rule and precision, with tests showing
  identical selection. Adds a manifest (GPU UUID, versions, revisions, prompt hash, tokens, sha256), resume, optional `attn_mass` /
  `true_rho` read-outs (proven not to change the selection; bit-identical traces on all 14 dev docs), multi-turn sessions, YaRN,
  sampling and page/K overrides. `collect/sources.py` covers RULER and SCBench.
- `kvpolicy_v2.py`: Random, ArkVale (sink/window pins; full version `simulate_arkvale` with forced pages, window rule, digests), InfLLM Eq. 1 and code `lru-s` (true mass or proxy), SIEVE, S3-FIFO
  (checked against independent references), lazy/frozen EMA, Holt, power-law kernel and the CO2 factorial. `kvpolicy.py` only gains a
  registry hook and an optional `attn=` argument; the existing classes are unchanged.
- `analysis_v2/`: S4 driver (sweep, iso-miss capacity, α curves, factorials, prediction metrics, paired bootstrap, noise),
  independent LRU/Bélády audit against `replay.py` (0 mismatches over all replays), 8B freeze, split builders, exploration, S7 driver
  and summaries.
- `gpu_bench_batch.py` + tests: the batched two-tier cache.
- New split files (fixed seeds, no reuse of longctx_v1 documents): `longctx_8b_v1`, `longctx64k_v1`, `ruler16k_v1`, `scbench_v1`.

## Deviations (all also in `docs/rerun/` and the final report)
1. **Order of S4/S5**: the S5 held-out collection ran *in parallel* with the CPU-only S4 analyses, after the replication
   preregistration was pushed. S4 output was never used for S5, and the frozen files were never touched.
2. **8B split**: the 16–32K GovReport/QMSum pool not used by longctx_v1 had only 25 documents (all from LongBench `gov_report_e`), so the
   floor was relaxed to 12K as the plan prescribes: 25 documents at 16–32K plus 10 at 12–16K. Held-out: 20 GovReport, 1 QMSum.
3. **64K**: 2 of 14 documents middle-truncated to 62K tokens (LongBench rule); YaRN factor 2; 6/14 stopped on the loop rule.
4. **SCBench**: contexts middle-truncated to 24K tokens to fit 32K with 5 turns (not specified in the plan).
5. **ArkVale baseline**: the full version is now run (S4.9, `docs/rerun/s4/09_arkvale_full.md`): forced sink/recent pages, the
   window rule, layers 0–1 unbudgeted and the cuboid-mean digest (plus the paper's other digests), following the official code
   line by line. Remaining differences, the same for every policy: 16-token pages (page- and token-count conversions both
   reported), the page being written is outside the cache, attention over the selected pages + tail instead of every resident
   page (needed for trace replay), cold start instead of ArkVale's prefill eviction, digest in float32.
6. **Power-law baseline**: implemented with an exponential-mixture approximation (max relative error 2.7 % at γ=1).
7. **Batched benchmark**: requests are cut to 16 384 tokens (first/last halves of the chat prompt); 8B batch 32 uses 12 288 tokens because
   only 25 8B documents reach 16K. The batched policy step is unfused (pessimistic for EMA).
8. **Results location on wulab2**: outputs were moved from `results_v2/` to an untracked `out/results_v2/` to avoid `git pull` conflicts
   (copied to the repository with rsync first; same bytes).
9. Concurrent collections on the same card during S6 (up to three processes). Traces are deterministic, so only the wall time is
   affected; all S7 measurements ran with no other process on the card.
10. `checks.py` reads `prompt_tokens` from the manifest when the npz lacks it (new traces store it in the manifest; later ones also in the npz).
11. **Post-hoc analyses (added in followup-v3, R9).** (a) On the 8B held-out, the InfLLM baselines with the true attention mass and the
    Spearman ρ were run after the evaluation with `analysis_v2.explore data/8b_heldout --only ...`, a command not in the
    preregistered list (the preregistration names them only as "also reported"); `results_v2/ext/8b_heldout_readouts/`. (b) The S5
    float16 tie analysis (`results_v2/heldout/ties.csv`) was also run after the evaluation without a preregistered command. Both
    are post-hoc, exploratory readings of fixed configurations; nothing was selected from them.
12. **Scope of the 8B result (R10).** All 35 8B documents stopped at the 1024-token cap, so the 8B confirmation covers only the
    first 1 023 decode steps, all inside the thinking phase. The (iv) threshold 0.15 was set after the 0.6B results (G ≈ 0.28) were
    known and before the 8B held-out was collected.
13. **Provenance (R11).** Manifests (without generated text), collector status files, logs, the executed run scripts and a re-run
    of the bit-identity checks are now in `results_v2/provenance/`; `runs/run_s6_8b_heldout.sh` is reconstructed after the fact.

No frozen file was changed; no PR was merged or closed; `main` and `efficient-policy` are untouched.

🤖 Generated with [Claude Code](https://claude.com/claude-code)

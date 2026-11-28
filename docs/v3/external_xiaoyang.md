# External results: Xiaoyang Wu (PR #9, PR #10)

**以下结果由 Xiaoyang Wu（GitHub `wxy20050403`）完成并报告，我们没有重跑，只核对了他提交的结果文件。** The user agreed to integrate them.
Source: his pull requests from a fork, fetched read-only (`git fetch origin pull/9/head:refs/remotes/pr/9 pull/10/head:refs/remotes/pr/10`);
nothing was checked out onto our branches, merged, or commented. His raw data files are **not** copied into this branch; they are
referenced by path and commit. Our check: `analysis_v3/xy_check.py`, run on wulab2 on files extracted from those refs, output
`results_v3/external_xiaoyang/check.json` (document bootstrap 2000, seed 0, as in his scripts).

**Summary.** (a) An independent replication on a third hardware type (Apple M1 Max, MPS) passes all three preregistered criteria
of the original held-out test (EMA G 0.280, 21/21, allocation + EMA ΔG +0.120) — consistent with our RTX 6000 Ada replication
(0.279) and the L4 original (0.281). (b) On new documents and a new task (NarrativeQA) G stays at 0.28 (exploratory). (c) A batched
transfer benchmark on an L4 shows that each saved miss saves ~5.3 µs and that LRU's transfer share of the step saturates at ~14 %
(B=96), which caps any policy's step speedup (Bélády +5.4 %, EMA +1.5 %): the same "transfer share sets the ceiling" picture as our
part E, measured differently. (d) On his own 20 short dev documents, adding the stage-1/2 rank term changes 2.9 % of eviction
decisions and saves 0.07 % of repeat misses, consistent with our earlier post-hoc finding that the rank term is negligible.

## (a) Apple M1 Max (MPS) replication — PR #9
Files: `docs/results/mps_dev/REPLICATION.md` (commit `9ff3ab1`), `docs/preregistration_mps.md` (preregistration, commit `1b6fe70`,
before any held-out capture), `docs/heldout_results_mps.md` (results, commit `29e59ee`); data `docs/results/mps_dev/`,
`docs/results/mps_heldout/misses.csv`, `splits/heldout_frozen_mps.json`. Hardware: Apple M1 Max, MPS, bf16, torch 2.15.0,
transformers 4.53.3, prefill in 1 024-token chunks. Samples: the 14 longctx_v1 dev documents (re-collected on MPS; generated tokens
differ from L4, so an independent trace set) and the 21 longctx_v1 held-out documents (re-collected on MPS, evaluated once with the
configuration frozen on MPS dev).

| | his report | our check | our results for comparison |
|---|---|---|---|
| dev, EMA G at B=96 (LODO) | 0.278 [0.275, 0.281] (text) | his `summary.csv` gives 0.278 [0.274, 0.280]; the dev check counts in his preregistration (LRU / EMA / Bélády 1 065 097 / 959 392 / 684 401) give 0.2777 — **checked (arithmetic); CI not recomputed** | 0.279 (RTX 6000 Ada dev, S3), 0.279 (L4 dev) |
| dev, allocation + EMA G at B=96 | 0.400 | **not checked** | 0.402 (RTX 6000 Ada dev) |
| held-out (i) EMA G at B=96 [CI] | 0.280 [0.276, 0.283] | **0.280 [0.276, 0.283] — checked** | 0.279 [0.275, 0.282] (S5), L4 0.281; third set 0.281 (F3) |
| held-out (ii) documents better than LRU | 21/21 | **21/21 — checked** | 21/21 |
| held-out (iii) allocation + EMA − EMA, ΔG [CI] | +0.120 [0.111, 0.128] | **+0.120 [0.111, 0.128] — checked** | +0.119 [0.112, 0.127] (S5) |
| held-out curve, EMA G at B = 72 / 80 / 128 / 256 | 0.256 / 0.275 / 0.265 / 0.235 | **same — checked** | 0.256 / 0.275 / 0.268 / 0.239 (S5) |
| held-out allocation + EMA G at B = 80 / 96 / 128 | 0.306 / 0.399 / 0.526 | **same — checked** | 0.306 / 0.398 / 0.524 (S5) |

Limits (his and ours): one model (Qwen3-0.6B), Quest selector, K=64, page 16; one collection on MPS; the held-out documents are the
team's longctx_v1 set (already evaluated on L4 and by us on RTX 6000 Ada), new only on this hardware. One held-out document was
resumed after a Mac sleep, as his preregistration allowed.

## (b) New documents and a new task — PR #9, same file, part F (exploratory)
His label: "单独报告、不判定成功" (reported separately, no success criterion); we treat it as exploratory evidence only.

| Group | documents | his EMA G [CI], wins | our check | allocation + EMA − EMA (his / check) |
|---|---|---|---|---|
| NarrativeQA (new task, 16.5–30K tokens) | 8 | 0.283 [0.278, 0.286], 8/8 | **0.283 [0.278, 0.286], 8/8 — checked** (`docs/results/v2_ood/misses.csv`) | +0.129 [0.107, 0.149] / **same** |
| GovReport > 32K, middle-truncated to 28–30K | 3 | 0.284 [0.277, 0.291], 3/3 | **0.284 [0.277, 0.291], 3/3 — checked** (`docs/results/v2_fresh/misses.csv`) | +0.096 [0.071, 0.113] / **same** |

Note: these documents are new to the team's 32K held-out protocol, but all 3 GovReport documents and 4 of the 8 NarrativeQA
documents are also in our exploratory 64K setting (`splits/longctx64k_v1.json`, at 32–64K tokens with YaRN, different prompts and
lengths, dev only), where the EMA gave 0.284. Nothing was selected on them in either study. Compare our task coverage: RULER 0.255,
SCBench 0.285, 64K 0.284 (replication-v2 S6).

## (c) Batched transfer benchmark — PR #9
File: `docs/batch_benchmark.md` (commit `475dcd1`), data `docs/results/batch_bench_l4/batch_bench.csv`. Hardware: Colab NVIDIA L4
(PCIe gen4 x16 as provisioned), torch 2.14.1+cu130, Triton 3.8.0; 21 MPS held-out documents' miss schedules replayed per batch
sequence; 256 timed steps. **Method caveat (his):** transfer (zero-copy gather of the missed pages) and compute (one Qwen3-0.6B decode
step + Quest scoring, synthetic inputs of the right shape) are timed separately with the profiler and **added**, not run end to end;
CPU launch time excluded; no overlap.

| B, batch | LRU transfer share of the step (his = check) | step speedup vs LRU: EMA / alloc + EMA / Bélády (his = check) | µs per page |
|---|---|---|---|
| 96, 1 | 6.8 % | +0.7 % / +0.9 % / +2.4 % | 6.0 |
| 96, 8 | 13.4 % | +1.4 % / +2.0 % / +5.1 % | 5.4 |
| 96, 32 | 14.0 % | +1.5 % / +2.1 % / +5.4 % | 5.3 |
| 96, 64 | 14.2 % | +1.5 % / +2.1 % / +5.4 % | 5.3 |
| 128, 32 | 8.8 % | +1.0 % / +1.9 % / +3.7 % | 5.4 |

**Checked:** every transfer share and speedup in his CSV is reproduced exactly from its raw time columns (max difference 0.0;
step = decode + scoring + transfer, speedup = LRU step / policy step − 1), and µs per page = transfer / pages.
Reading next to ours: his decomposition gives the ceiling directly — a policy can save at most (its saved misses) × 5.3 µs, i.e. at
most the transfer share times its share of the miss gap. Our part E measures the whole decode step end to end on the fixed batched
path (`docs/v3/e_systems.md`), and replication-v2 S7 measured a fetch phase of ~18 % of the step at batch 16 on an RTX 6000 Ada
(unfused path). The two approaches agree that transfer is a minority of the step on a PCIe GPU, so the policy's byte saving can
only become a few percent of throughput; his numbers are an explanation of that ceiling, not a replacement for our end-to-end
measurement.

## (d) How often the rank term changes eviction decisions — PR #10
File: `docs/results/solo_rank_dev/README.md` (commit `b867330`), data `docs/results/solo_rank_dev/event_counts.csv`,
`decision_summary.csv`, `reuse_summary.csv`. Samples: his personal 20 dev documents (GovReport and MultiFieldQA-en, 4–16K tokens,
127 decode steps), 28 layers, K=64, B=128. Mechanism analysis on the same documents used to choose the weights (no held-out).

| | his report | our check |
|---|---|---|
| decisions changed by adding the rank term, R+F (β=2) | 2.93 % [2.41 %, 3.45 %] (1 053 of 35 975) | **1 053 / 35 975 = 2.93 % — checked; CI not recomputed** |
| same with LRU (β=0) | 3.22 % [2.67 %, 3.77 %] | **3.22 % — checked; CI not recomputed** |
| repeat misses in independent replays, R+F vs R+F + rank | 82 088 vs 82 029 (−59, 0.072 %) | **−59 (0.072 %) — checked** |
| LRU vs LRU + rank | 82 274 vs 82 209 (−65) | **−65 — checked** |
| frequency-matched 8-step reuse, top-16 vs bottom-16 rank (selected pages) | +8.41 pp [8.06, 8.74] | **not checked** |

Comparison: stage 1–2's post-hoc clean rank increment on the 40-document held-out was 213 repeat misses (0.14 %) (docs/STATUS_REPORT.md
§4.7), and our B7 shows why rank among the selected pages carries almost nothing for eviction (G 0.016 for an EMA of that rank).
His measurement adds the decision-level view: rank does predict reuse (+8 pp), but it changes only ~3 % of evictions, and most of
those changes do not matter for the total.

## Where these results are cited
`docs/v3/thesis_v3.md` (replication across hardware; the system ceiling), `docs/v3/pr_description.md`, `docs/v3/e_systems.md`,
`README.md` and `docs/STATUS_REPORT.md` (hardware coverage). Always as "Xiaoyang Wu, PR #9 / #10".

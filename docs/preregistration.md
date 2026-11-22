# Preregistration: held-out evaluation

This document fixes the analysis plan before any held-out prompt is collected. It is committed together with the machine-readable configuration `splits/preregistration.json` and the scripts that execute it (`evaluate.py`, `truncate_trace.py`, `longbench_score.py`). After this commit the plan does not change. Any deviation will be stated explicitly in the report.

## Data

- **Split.** `splits/longbench_v1.json` (sha256 `b8252589…d99f`): 60 prompts, 15 per task (MultiFieldQA-en, GovReport) × length band (4–8K, 8–16K Qwen tokens), divided 20 dev and 40 held-out by source document.
- **Dev.** All 20 dev prompts were collected on an NVIDIA L4 (bf16, torch 2.14.1+cu130, transformers 4.53.3). Trace sha256 `a96cb40d…eb5f`. All tuning below used only these traces.
- **Held-out.** The 40 held-out prompts will be collected once, with `SUBSET = "heldout"` and `DENSE_REFERENCE = True` in `notebooks/first_trace.ipynb`.
  - The protocol is identical to dev: same selector, K = 64, 16-token pages, greedy decoding, batch 1, generation stops at 128 tokens or EOS, no truncation at collection.
  - Held-out must be collected on the same GPU model as dev (NVIDIA L4). Quest selections are deterministic on one GPU model but differ between A100 and L4 for the same prompt, so mixing GPU models would confound dev and held-out.
- **Pilot data excluded.** The earlier eight-document dev pilot (A100) is not used for tuning or evaluation.

## Tuning on dev (done; results frozen in the JSON)

Proposal rule: at B = 128, choose δ, β ∈ {0, 1, 2, 4, 8} by dev miss rate, with ties going to the smaller value (smaller δ first, then smaller β). R+F tunes β with δ = 0; R+rank tunes δ with β = 0 the same way.

Dev misses at B = 128 (rows δ, columns β):

| δ \ β | 0 | 1 | 2 | 4 | 8 |
|---|---|---|---|---|---|
| 0 | 187,866 | 187,733 | 187,689 | 187,672 | 187,951 |
| 1 | 187,787 | 187,662 | 187,616 | 187,605 | 187,890 |
| 2 | 187,778 | 187,659 | 187,611 | 187,606 | 187,893 |
| 4 | 187,729 | 187,634 | 187,600 | 187,593 | 187,879 |
| 8 | 187,739 | 187,622 | **187,568** | 187,573 | 187,856 |

Chosen parameters:
- Full policy: **δ = 8, β = 2**.
- R+F: **β = 4**.
- R+rank: **δ = 4**.

**Logistic regression** (from `gap_analysis.py`):
- Fit once on all 20 dev documents, using the candidates at LRU eviction decisions at B = 128.
- Label: reselected within 16 steps. 16 trace-derived features; 300,000 sampled rows; seed 0; L2 = 1e-3.
- Coefficients, means and standard deviations are frozen in the JSON.
- At eviction, the candidate with the lowest predicted reuse leaves first, with ties broken in LRU order.

## Policies and budgets

| Name | Policy |
|---|---|
| `fifo` | FIFO |
| `lru` | LRU (same-step hits ranked above fills, then page ID) |
| `rank_only` | Evict the worst last rank first |
| `rf` | Score policy, δ = 0, β = 4 |
| `rrank` | Score policy, δ = 4, β = 0 |
| `full` | Score policy, δ = 8, β = 2 |
| `logistic` | Frozen logistic regression |
| `belady` | Offline oracle (farthest next use) |

Budgets: B = 128 (primary) and B = 256, i.e. B/K ∈ {2, 4}. Every replay starts empty, protects the current selection and admits every miss.

## Metrics

All counts are pooled over documents and layers before division.
- **Repeat misses and total misses**, together with their rates.
- **G** = (repeat misses of LRU − repeat misses of π) / (repeat misses of LRU − repeat misses of Bélády).
- **H2D bytes per generated token** = total misses × 65,536 B / total sparse decode steps. One page per layer is 16 tokens × 8 KV heads × 128 dims × (K, V) × 2 bytes. It is reported in MiB per token.
- **Intervals.** Percentile bootstrap over documents: 2,000 replicates, seed 0. The same resampled document sets are used for every policy within a budget.

## Success criteria (B = 128, held-out)

1. **S1.** The full policy has at least 10% fewer repeat misses than LRU (pooled point estimate). The 95% bootstrap CI of the reduction is reported alongside.
2. **S2.** The repeat misses saved by the full policy relative to R+F have a paired document-bootstrap 95% CI that lies entirely above zero.

S1 alone does not show that rank helps, because frequency alone could meet it. S2 isolates the rank term.

## H1

- **Samples:** `selected`, `candidate` and `lru` (LRU at B = 128), as implemented in `analysis.py`.
- **Horizons:** 1, 2, 4 and 8. Only steps followed by 8 observed steps are used.
- **Effect:** top minus bottom quartile within frequency × recency strata, weighted by stratum size.
- **Inference:** percentile bootstrap that resamples documents (40 clusters), 1,000 replicates, seed 0. Also reported: raw and recency-only differences, the top-quartile share, and results by task and by band.

## Groups

Every policy table is also reported by task and by length band, with document-bootstrap intervals within each group.

## MultiFieldQA rule

- **Primary analysis:** the traces as collected (128 tokens or EOS).
- **Secondary analysis:** `truncate_trace.py` cuts each MultiFieldQA request after the steps that produce its first answer, up to the first generated newline. The same policies, fixed parameters and coefficients are then re-evaluated.

## LongBench sanity check

- For each held-out prompt, score the sparse reference generation and a dense-attention greedy generation of the same length cap with LongBench's metrics: `qa_f1_score` for MultiFieldQA-en, ROUGE-L F for GovReport, maximum over reference answers, ×100.
- Report means by task and band, plus the paired sparse − dense difference with a document bootstrap.
- This checks that the sparse reference run is a reasonable stand-in for dense decoding. It is not a success criterion.

## Exact commands (run once on held-out)

```bash
python3 evaluate.py results/heldout --out results/heldout/eval
python3 truncate_trace.py results/heldout
python3 evaluate.py results/heldout --trace trace_truncated.jsonl --out results/heldout/eval_truncated
python3 longbench_score.py results/heldout --out results/heldout/longbench
```

`evaluate.py`, `truncate_trace.py` and `longbench_score.py` need numpy and transformers for the tokenizer; `longbench_score.py` also needs rouge 1.0.1. The same commands were dry-run on dev (`results/dev_full/`) before this commit. The dev outputs give the dev side of the dev-versus-held-out comparison.

## What counts as a deviation

Any of the following will be reported as a deviation:
- a change to policies, parameters, coefficients, budgets, metrics, bootstrap settings, criteria or the MultiFieldQA rule after this commit;
- re-running the held-out evaluation after inspecting its results;
- collecting held-out on a different GPU model.

Bug fixes that change held-out numbers would also be deviations, and both the before and after numbers would be reported.

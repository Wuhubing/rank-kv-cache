# Long-trace pilot (dev only, one document plus probes)

Setup: Qwen3-0.6B (rev c1899de), bf16, Colab L4. Quest-style selector unchanged from
`notebooks/first_trace.ipynb` (min/max bounds, query heads averaged, 16-token pages, K=64).
Traces are captured on the GPU into int16 rankings `[steps, layers, K]` and float16 page scores
`[steps, layers, pages]` (all historical pages; ~97 MB per 1K-step, 26K-token document; npz).

## Task and stop rule
- Plain-prompt answers (no chat template) never emit EOS. After about 250 tokens the model
  restarts the answer verbatim, which would inflate hit rates. Rejected.
- Chat template with thinking enabled, asking a specific QMSum query: clean EOS at 304 steps. Too short.
- **Chosen:** chat template, thinking enabled, a request to summarise the *whole* meeting or report
  "in at least eight paragraphs". Four probe documents gave 741, 794, 832 and 873 steps, all ending
  at EOS (about 300 thinking tokens), with no loops. Long decoding with thinking is also the regime
  where decode-time KV traffic matters.
- Stop rule: EOS, 1024 tokens, or the newest 32-gram already occurring in the output (catches verbatim
  restarts and loops). None of the probes triggered it.
- Candidate pool (prompt 24–32K Qwen tokens): QMSum 22 and GovReport 4. MultiNews never reaches
  24K. NarrativeQA is short-answer QA.

## Result (qmsum/cbbd3e6a…, 26,120-token prompt, 741 steps)
| B (pages) | B/K | LRU miss | Bélády miss | gap |
|---|---|---|---|---|
| 72 | 1.12 | 11.42% | 8.30% | 3.12 pp |
| 80 | 1.25 | 8.60% | 5.78% | 2.83 pp |
| 96 | 1.50 | 5.70% | 3.54% | 2.15 pp |
| 128 (2K tok) | 2 | 3.32% | 1.98% | 1.35 pp |
| 256 (4K tok) | 4 | 1.27% | 0.92% | 0.35 pp |

Repeat-miss rate at B=128: LRU 2.55% vs Bélády 1.19%. Bélády removes about half of LRU's repeat
misses, but in absolute terms that is small.

- Page granularity does not change this. With the same 2K-token budget, page 8 gives a gap of
  1.53 pp and page 4 gives 1.75 pp.
- The headroom is concentrated in a few layers. At B=128, LRU/Bélády are L11 12.8/7.5%,
  L16 8.0/4.7%, L25 9.7/5.7% and L24 7.4/4.2%. Layers 0–5, 7–10 and 27 are each ≤1.4%.

**Conclusion:** a 32K context and ~750-step generation do not restore a gap of several points at
B/K=2 ("twice the Top-K size", as the course problem states). Our interpretation is a granularity
difference rather than a different B. HiSparse caches individual tokens. We cache 16-token pages,
and page-level selections are more stable across adjacent decode steps, so at B/K=2 the headroom is
naturally smaller. This is a finding. B keeps its definition (pages per layer, as a multiple of K);
we do not redefine it to match the paper. The page-size probe (16 → 8 → 4 slightly widens the gap)
points the same way but cannot isolate the effect at token granularity. Single-document pilot.

**Decision (team):** primary budget B/K=1.5 (B=96), full curve B/K ∈ {1.125, 1.25, 1.5, 2, 4}.
This is frozen in `splits/longctx_v1.json`.

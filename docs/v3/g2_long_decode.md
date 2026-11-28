# G2 (exploratory): Qwen3-8B decoded to the end

**Conclusion: the EMA's G does not depend on the 1 024-token cap. With the cap raised to 4 096 tokens every 8B dev document ends
at EOS (1 503–2 018 decode steps), and G is 0.278 [0.274, 0.282] at B=96 (14/14), the same as with the cap (0.272); in the answer
after thinking it is 0.281, in the thinking phase 0.270, and in the last quarter of the decode 0.290.** This answers R10's question:
the 8B result is not an artefact of looking only at the first 1 023 steps (which already contained the end of thinking and the
start of the answer).

Data: `runs/run_g2_collect.sh`, `collect.long --split splits/longctx_8b_v1.json --subset dev --max-new 4096` (wulab2 card 0,
GPU-a74391df-5c27-152f-d0c0-6b86f8ea07d3, 21:38–22:04Z, 2.3 GB, `data/8b_dev_long`). The first 1 023 generated tokens of every
document are identical to the capped traces (14/14). Thinking ends at step 458–843. Manifests:
`results_v3/provenance/8b_dev_long_manifests.jsonl`. Replays: `analysis_v3.run base` (`results_v3/g2/base/`, audit 4 032 streams,
0 mismatches) and `analysis_v3/instrument.py` (`results_v3/g2/inst/`). Exploratory: dev documents, no preregistration.

| | capped (1 023 steps, `results_v3/b1/b1_8b/`) | full decode (to EOS) |
|---|---|---|
| decode steps per document | 1 023 | 1 503–2 018 |
| EMA G, B=96 (LODO) | 0.272 [0.268, 0.276] | **0.278 [0.274, 0.282]**, 14/14 |
| EMA G, B=128 | 0.256 | 0.261 [0.256, 0.267] |
| percentile EMA, B=96 / 128 | 0.263 / 0.241 | 0.270 / 0.249 |
| current score, B=96 / 128 | 0.221 / 0.172 | 0.229 / 0.181 |

By phase (full decode, B=96, EMA α 0.5; LRU misses per step and layer):
| Phase | layer-steps | LRU misses / step·layer | EMA G [CI] | current score G |
|---|---|---|---|---|
| first quarter | 230 580 | 3.13 | 0.271 [0.262, 0.279] | 0.221 |
| middle half | 461 628 | 2.79 | 0.275 [0.272, 0.278] | 0.226 |
| last quarter | 230 904 | 2.41 | 0.290 [0.284, 0.297] | 0.244 |
| thinking | 323 352 | 2.84 | 0.270 [0.264, 0.275] | 0.219 |
| answer | 599 760 | 2.74 | 0.281 [0.277, 0.285] | 0.234 |

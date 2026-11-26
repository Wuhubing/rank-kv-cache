# B5: time structure — is G the same early and late, while thinking and while answering?

**Conclusion: yes. The EMA's G is flat across the decode: 0.275 / 0.281 / 0.276 in the first quarter, middle half and last quarter
on 0.6B dev, and 0.276 (thinking) vs 0.280 (answer); on 8B 0.272 / 0.269 / 0.276 and 0.270 vs 0.274.** The miss rate itself
falls as decoding proceeds on 0.6B (LRU 4.7 → 2.9 misses per step and layer), but the policy closes the same share of the gap
throughout. The 8B result covers the end of thinking and the start of the answer within the first 1 023 steps; the full-length
decodes are in G2 (`docs/v3/g2_long_decode.md`).

Source: `analysis_v3/instrument.py` (per-step misses from `kvpolicy_v3.simulate_steps`; phase boundary = the step that emits
`</think>`, from the manifest token list), `analysis_v3/inst_summary.py`; `results_v3/inst/{dev,8b}/phase.csv.gz`,
`summary.json`. B=96, EMA α 0.5 and current score; G per phase with LRU and Bélády of the same steps; CI = document bootstrap.

| Phase | 0.6B steps | 0.6B LRU misses / step·layer | 0.6B EMA G [CI] | 0.6B current score G | 8B steps | 8B LRU misses / step·layer | 8B EMA G [CI] | 8B current score G |
|---|---|---|---|---|---|---|---|---|
| first quarter | 75 292 | 4.74 | 0.275 [0.270, 0.281] | 0.237 | 128 520 | 3.30 | 0.272 [0.261, 0.282] | 0.219 |
| middle half | 150 836 | 3.90 | 0.281 [0.276, 0.286] | 0.245 | 258 048 | 2.71 | 0.269 [0.264, 0.273] | 0.219 |
| last quarter | 75 572 | 2.91 | 0.276 [0.266, 0.286] | 0.233 | 129 024 | 3.00 | 0.276 [0.271, 0.281] | 0.227 |
| thinking | 122 584 | 4.15 | 0.276 [0.272, 0.280] | 0.236 | 323 352 | 2.84 | 0.270 [0.264, 0.275] | 0.219 |
| answer | 179 116 | 3.67 | 0.280 [0.274, 0.287] | 0.244 | 192 240 | 3.08 | 0.274 [0.269, 0.278] | 0.224 |

(Steps are layer-steps summed over documents. On 0.6B dev 41 % of the steps are thinking, on 8B dev 63 %; the 8B traces stop
at the 1 024-token cap, 180–565 steps into the answer.)

Error rates of the EMA's evictions against Bélády (B3) are also similar in both phases (0.6B 0.63 thinking vs 0.61 answer;
8B 0.64 vs 0.59).

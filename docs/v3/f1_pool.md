# F1: document pool for a third held-out set

**Conclusion: enough unused documents exist; F can be done with 21 documents for both models.** After removing every document
of the six used splits, 2 262 distinct documents of the original GovReport and QMSum datasets have a 16–32K-token prompt with
both the Qwen3-0.6B and the Qwen3-8B tokenizer. The first 21 in hash order (all GovReport) form `splits/longctx_v3.json`.

Source: `analysis_v2/survey_v3.py` (run on wulab2, CPU only, prompt lengths only, no trace collected; log
`results_v3/f1_pool/survey.json`).

| Step | Count |
|---|---|
| Distinct GovReport documents (`ccdv/govreport-summarization@4e21184e`, train + validation + test, field `report`) | 19 445 |
| Distinct QMSum transcripts (`pszemraj/qmsum-cleaned@d926f9cc`, `no-prefix`, train + validation + test, field `input`) | 233 |
| Removed by a cheap character-length prefilter (outside 30K–250K characters; a 16–32K-token prompt is ~60–150K) | 6 057 |
| Prompt outside 16 000 ≤ n < 32 000 tokens (0.6B tokenizer; the 8B tokenizer gave the same counts for every candidate) | 11 294 |
| In range | 2 260 GovReport + 67 QMSum |
| Removed by de-duplication against the used documents (identical context hash, or > 5 % of 8-word shingles found in a used document) | 55 + 10 |
| **Candidates** | **2 205 GovReport + 57 QMSum = 2 262** |

Used documents checked: longctx_v1 (35), longctx_8b_v1 (35), longctx64k_v1 (14), ruler16k_v1 (14), scbench_v1 (14), longbench_v1
(60); 65 candidates were copies of used LongBench GovReport/QMSum documents (shingle overlap up to 100 %). Largest shingle
overlap among the remaining candidates 4.9 %, among the 21 taken 2.6 %.

The 21 documents: hash order sha256("longctx_v3:" + rid); 16 102–28 278 prompt tokens (same with both tokenizers), 19 at
16–24K and 2 at 24–32K. All GovReport (QMSum is 2.5 % of the pool and none falls in the first 21). They are all held-out;
nothing in them is used for selection or tuning, and their traces are collected only after the preregistration
(`docs/preregistration_v3.md`, `splits/heldout_frozen_v3.json`) is committed and pushed (F2). The collector reads them through
the new `parquet` source of `collect/sources.py` (same prompt construction as LongBench).

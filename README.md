# Can Selection Rank Improve KV Cache Replacement?

A single-GPU study of HiSparse-style hierarchical KV caching (MIT 6.5940, Fall 2026, Project 10).

We test whether a page's sparse-attention top-K selection rank predicts reuse beyond
recency and frequency, using Qwen3-0.6B traces on LongBench (MultiFieldQA-en, GovReport).

## Team

Jiayu Li, Mingxi Wang, Weibing Wang, Xiaoyang Wu, Jimmy Yu

## First pilot

1. In [Google Colab](https://colab.research.google.com/), select **File → Upload notebook** and choose [first_trace.ipynb](notebooks/first_trace.ipynb).
2. Select **Runtime → Change runtime type → GPU** and **Runtime → Run all**. The notebook includes the replay functions and downloads pinned public model and data revisions. It uses BF16 on supported GPUs and FP16 otherwise.
3. Download the ZIP from the final cell. It contains `trace.jsonl`, `manifest.json`, `comparison.csv` and `comparison.png`. The same files remain under `results/` while the session is alive.

The pilot takes the first GovReport document long enough for a 4,096-token prefix-based summarization prompt. Dense prefill produces token 1. Subsequent tokens come from greedy sparse decoding with the full KV store on device. With a 64-token cap, at most 63 sparse steps are logged. A step selects 64 completed 16-token historical pages per layer and also attends to the pinned page containing the current query token. Each request uses one unpadded sequence. The manifest records the exact source document, tokenizer/model revision, token-ID hash, GPU, precision and generated IDs. Reserve this pilot source for development.

## Trace format (v1)

Use UTF-8 JSON Lines, one object per request, layer and decode step. For example:

```json
{"request_id":"toy","layer_id":0,"decode_step":0,"ranked_page_ids":[7,2,4]}
```

| Field | Meaning |
| --- | --- |
| `request_id` | Nonempty string identifying one generation run. Use different IDs for independent generations. |
| `layer_id` | Zero-based nonnegative layer index. Each request and layer has its own initially empty cache. |
| `decode_step` | Integer starting at 0, contiguous and increasing within its stream. Streams can be interleaved. |
| `ranked_page_ids` | Ordered, unique, nonnegative page IDs, best first. Index 0 means rank 1. An empty list is valid when no historical page exists. |

For the Qwen pilot, token positions and physical-page boundaries are zero-based. Historical page ID `p` covers positions `16*p` through `16*p+15`. The page holding the current token is pinned through that query. It becomes eligible starting with the next page. Its accesses and storage are excluded from historical-page counters and budget. Decode scores use per-coordinate extrema of post-RoPE keys, summed signed query bounds and mean over query heads, with smaller page ID first on a score tie. The separate manifest carries the page size and selector settings. Handwritten schema examples can use smaller K.

## Replay locally

From the repository root, run the unit tests with Python 3.10 or newer. Add Matplotlib for the plotting command.

```bash
python3 -m unittest discover -s tests -v
python3 -m pip install matplotlib
python3 replay.py examples/handwritten.jsonl --budgets 2 3 4 --out results/handwritten
```

For a downloaded Colab trace:

```bash
python3 replay.py path/to/trace.jsonl --budgets 64 128 256 --out results/govreport
```

Both policies observe the same set simultaneously, protect that entire selection, admit every miss and evict only as many unselected residents as necessary. LRU prefers the most recent step, then same-step hits over fills, then smaller page ID. Bélády evicts the resident with farthest next use, treating no future use as infinity and evicting smaller ID on a tie. Only the oracle sees future steps.

The left panel divides total misses by all selected historical page accesses. The right divides repeat misses by repeat accesses. A first selection is always compulsory within its request/layer stream. An undefined rate has an empty CSV field. Counts are pooled across streams before division. At the primary budget of 128, subtract the oracle miss rate from the LRU miss rate to measure the available headroom in percentage points. Store the raw file to replay future rank policies on identical selections.

**Handwritten check.** The eight accesses select four distinct pages. At budgets 2, 3 and 4, LRU has 5, 5 and 4 total misses. Bélády has 5, 4 and 4. The unit suite also checks protected selections, tie behavior, stream isolation and an independently enumerated optimum.

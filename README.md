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

The pilot takes the first GovReport document long enough for a 4,096-token prefix-based summarization prompt. Dense prefill produces token 1. Subsequent tokens come from greedy sparse decoding with the full KV store on device. With a 128-token cap, at most 127 sparse steps are logged. A step selects 64 completed 16-token historical pages per layer and also attends to the pinned page containing the current query token. Each request uses one unpadded sequence. The manifest records the exact source document, tokenizer/model revision, token-ID hash, GPU, precision and generated IDs. Reserve this pilot source for development.

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

From the repository root, run the unit tests with Python 3.10 or newer. The replay and analysis scripts need only the standard library. Matplotlib is optional and adds the PNG figures; without it, the CSV files are still written.

```bash
python3 -m unittest discover -s tests -v
python3 -m pip install matplotlib
python3 replay.py examples/handwritten.jsonl --budgets 2 3 4 --out results/handwritten
```

For a downloaded Colab trace:

```bash
python3 replay.py path/to/trace.jsonl --budgets 64 128 256 --out results/govreport
```

All policies observe the same set simultaneously, protect that entire selection, admit every miss and evict only as many unselected residents as necessary. Choose them with `--policies` (default `lru belady`):

| Name | Evicts first | Ties |
| --- | --- | --- |
| `lru` | Oldest last selection step | Fills before same-step hits, then larger page ID |
| `fifo` | Oldest admission; hits do not refresh | Larger page ID |
| `rank` | Worst rank at its last selection (rank 1 is best) | LRU order |
| `rf` | Lowest score with `delta=0`: recency plus frequency | LRU order |
| `rrank` | Lowest score with `beta=0`: recency plus rank | LRU order |
| `full` | Lowest score with both weights | LRU order |
| `score:D:B` | Lowest score with explicit weights, for example `score:2:1` | LRU order |
| `belady` | Farthest next use; no future use counts as infinity | Smaller page ID |

The score is `tau + delta * [rank <= ceil(K/4)] + beta * min(f, 4)`:
- `tau` is the page's last selection step and `rank` is its rank at that step.
- `f` counts the selections in the eight steps before the current one.
- `K` is `--top-k`, defaulting to the widest selection in the trace.
- `--delta` and `--beta` are in decode steps and feed `rrank`, `rf` and `full`.

With both weights at zero, the score policy reproduces LRU eviction step by step. Only the oracle sees future steps. For example:

```bash
python3 replay.py path/to/trace.jsonl --budgets 128 256 \
  --policies lru fifo rank rf rrank full belady --delta 4 --beta 2 --out results/govreport
```

The left panel divides total misses by all selected historical page accesses. The right divides repeat misses by repeat accesses. A first selection is always compulsory within its request/layer stream. An undefined rate has an empty CSV field. Counts are pooled across streams before division.

`comparison.csv` has one row per budget and policy. Its columns are `budget`, `policy`, `delta`, `beta` (empty for policies without weights), `top_k`, `accesses`, `misses`, `compulsory_misses`, `repeat_misses`, `repeat_accesses`, `miss_rate` and `repeat_miss_rate`. At the primary budget of 128, subtract the oracle miss rate from the LRU miss rate to measure the available headroom in percentage points. Store the raw file to replay future rank policies on identical selections.
**Handwritten check.** The eight accesses select four distinct pages. At budgets 2, 3 and 4, LRU has 5, 5 and 4 total misses. Bélády has 5, 4 and 4. The unit suite also checks protected selections, tie behavior, stream isolation, an independently enumerated optimum, step-by-step agreement of the zero-weight score with LRU, and the oracle as a lower bound for every policy.

## Rank predictiveness (H1)

```bash
python3 analysis.py path/to/trace.jsonl --by task layer-group --out results/h1
```

The script records one observation per selected page at decode step t:
- its rank at t (its most recent rank);
- recency, the number of steps since its previous selection, or `new` for a first selection;
- frequency, its selections in steps t−8 to t−1;
- whether it is selected again within 1, 2, 4 and 8 steps.

Only steps followed by `max(--horizons)` observed steps are used, so every horizon is measured on the same observations. Short traces print a message and need smaller `--horizons`.

The effect compares the top quartile (rank ≤ ⌈n/4⌉) with the bottom quartile (rank > n − ⌈n/4⌉) of each step's n selected pages:
- It is computed within strata of frequency × recency bin (`new`, 1, 2, 3–4, 5–8, >8).
- Strata are weighted by size, and only strata containing both quartiles count.
- The 95% interval is a percentile bootstrap (`--bootstrap 1000`, `--seed 0`) that resamples whole requests, meaning source documents.
- With a single request, the interval is reported as n/a. `--cluster stream` resamples (request, layer) pairs instead, which is useful for a pilot but ignores correlation across layers.

Outputs in `--out`:
- `h1_effects.csv` has one row per group and horizon. Its columns are `group` (`all`, `task <request_id prefix>` or `layers a-b`), `horizon`, `observations`, `clusters`, `strata_used`, `effect` (stratified top minus bottom reuse probability), `ci_low`, `ci_high`, and the unadjusted `raw_top` and `raw_bottom` rates.
- `rank_curve.csv` gives `horizon`, `rank_decile` (1 is best), `observations` and `reuse_rate`.
- `features.csv` is written with `--features`. It has one row per observation: `request_id`, `task`, `layer_id`, `decode_step`, `page_id`, `rank`, `n_selected`, `quartile`, `recency`, `recency_bin`, `frequency` and `reuse_h`.
- `h1.png` plots reuse by rank decile and the stratified effect against horizon.

**Caveat:** rank is relative within a step. Because every step selects exactly K pages, a page outranked by frequent competitors can show lower reuse through competition alone, even when its own propensity depends only on frequency. The synthetic tests in `tests/test_analysis.py` show this effect with a small page pool.

## Related work

[docs/related_work_matrix.md](docs/related_work_matrix.md) compares ArkVale, HiSparse, Quest, H₂O and PARROT. It records which claims we can and cannot make, and which items remain unverified.

# Post-hoc validation of the selection-rank result

This document is exploratory. It does not change the preregistered held-out analysis or
its success criteria.

## Question

The preregistered full policy used `delta=8, beta=2`, whereas its tuned R+F baseline
used `delta=0, beta=4`. Their difference therefore does not isolate rank at a fixed
frequency weight. The first robustness check compares `delta>0` with `delta=0` at the
same beta.

## Evidence already recoverable from committed summaries

The complete 5-by-5 development grid is preserved in
`docs/results/dev_full/tuning.csv`. For every beta, the best positive delta improves
repeat misses by less than 0.18%:

| Fixed beta | Delta=0 repeat misses | Best delta | Repeat misses saved | Reduction |
|---:|---:|---:|---:|---:|
| 0 | 78,303 | 4 | 137 | 0.175% |
| 1 | 78,170 | 8 | 111 | 0.142% |
| 2 | 78,126 | 8 | 121 | 0.155% |
| 4 | 78,109 | 8 | 99 | 0.127% |
| 8 | 78,388 | 8 | 95 | 0.121% |

The preregistered held-out table contains one matched-beta comparison: R+rank
(`delta=4, beta=0`) versus LRU (`delta=0, beta=0`). It saves 265 of 151,478 repeat
misses, or 0.175% with a document-bootstrap interval of [0.115%, 0.232%]. This almost
exactly reproduces the development effect at the same parameters.

These available checks strengthen the narrow conclusion that the implemented
top-quartile last-rank bonus has a real but practically negligible effect. They do not
show that every possible representation of selection rank is ineffective.

## New validation implementation

`rank_validation.py` produces four files from an original trace:

- `matched_beta.csv`: full-policy miss counts and document-bootstrap intervals for
  every delta at each fixed beta;
- `matched_beta_by_doc.csv`: the corresponding document-level counts;
- `decision_audit.csv`: one-step counterfactual victim changes on the delta=0 cache
  trajectory;
- `decision_audit_by_doc.csv`: document-level audit counts.

Run the formal setting with:

```bash
python3 rank_validation.py results/heldout/trace.jsonl --budgets 128 \
  --deltas 0 1 2 4 8 --betas 0 1 2 4 8 \
  --out results/heldout/rank_validation
```

The repository does not currently contain the original dev or held-out JSONL traces,
so the complete matched-beta held-out table and decision audit cannot be regenerated
from the public checkout alone. They should be run on the archived trace whose SHA-256
is recorded in the run manifest. No new model generation is required.

For a stricter H1 robustness check, exact recency strata are now supported:

```bash
python3 analysis.py results/heldout/trace.jsonl --samples lru --lru-budget 128 \
  --recency-strata exact --out results/heldout/h1_exact
```

Exact strata can be sparse, so both the number of usable strata and the interval must
be reported. The preregistered binned-recency result remains the primary analysis.

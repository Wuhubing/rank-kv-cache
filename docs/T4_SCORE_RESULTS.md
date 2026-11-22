# Trace v2 T4 score-magnitude experiment

**Status:** exploratory, eight `dev_collection` documents, Tesla T4, BF16, chunked
prefill (512 tokens), 127 sparse decode steps per document. This is not part of the
preregistered L4 held-out analysis.

The run completed 8/8 documents and produced 28,448 request--layer rows. The Trace v2
SHA-256 is `c19c0eba9182f85d5c6773b8c07f1f483de7ad1b255cd33c9e80df335c11953a`.

## Score magnitude does not improve replacement

The score-margin policy adds `gamma * last_margin` to effective recency, where margin
is normalized between the Kth selected score (zero) and the best selected score (one).
Every gamma is compared with gamma zero at the same frequency weight beta.

The best of 30 configurations is `beta=8, gamma=8`: it saves 35 repeat misses, or
0.102%, with a document-bootstrap interval of [-0.226%, 0.431%]. Three of eight
documents become worse. At beta zero, every positive gamma is worse than LRU; its best
configuration changes repeat misses by -0.123%.

Therefore, retaining the magnitude discarded by ordinal rank does not rescue the
policy. The point estimate is smaller and less stable than the last-rank bonus.

## Matched-beta rank result replicates on T4

The best rank configuration is `beta=0, delta=4`: it saves 42 repeat misses, or
0.123% [0.032%, 0.226%]. The best configuration at every beta lies between 0.103% and
0.123%. This matches the earlier A100 development and L4 held-out conclusion that the
top-quartile last-rank bonus has a real but practically negligible effect.

## Mechanism

On the matched delta-zero cache trajectories, top-quartile pages are only about
1.06--1.10% of eviction candidates. Depending on beta and delta, the rank bonus changes
only 2.3--3.0% of eviction events. For `beta=0, delta=4`, it protects 525 top-quartile
victims across 14,768 eviction events, but the full replay ultimately saves only 42
repeat misses.

This explains the gap between prediction and policy value: a page's last rank predicts
reuse, but high-ranked pages rarely enter the eviction population, and many one-step
victim improvements do not survive later cache-state interactions.

## Exact-recency robustness

Restricting H1 to LRU-resident candidates and stratifying by exact recency rather than
the preregistered bins still yields positive effects:

| Reuse horizon | Top-minus-bottom effect | 95% document-bootstrap interval |
|---:|---:|---:|
| 1 | 4.20 pp | [2.67, 6.16] pp |
| 2 | 6.82 pp | [4.37, 9.78] pp |
| 4 | 10.48 pp | [6.72, 15.46] pp |
| 8 | 14.70 pp | [9.65, 21.00] pp |

Only 0.776% of observations have a top-quartile last rank. Thus the predictive signal
is not an artifact of coarse recency bins, but it remains rare in the decision set.

## Conclusion

The experiment supports a stronger, still scoped negative result:

> In this Quest-style setup, historical selection importance is predictive of reuse,
> but neither ordinal last rank nor selected-score magnitude provides practically
> meaningful cache-replacement improvement. The limitation is primarily actionability
> at eviction time, not loss of magnitude in the rank representation.

The next technically distinct hypothesis would require current-query scores for all
resident candidates, including pages outside the selected top K. That is different from
historical rank and would need a larger Trace v3 representation. Given the sub-0.2%
effects here, it should be pursued only if the project needs a new exploratory direction;
the present negative result is already well supported.

# S4.6 The α curve from "current score" to "frequency", three signals (dev)

**Conclusion.** The score EMA and its two "rivals" are points on one curve: α=0 is the current score (ArkVale /
FlashOffloading "Score"), and α→1 approaches a long-term frequency-like average. G rises from 0.240 at α=0 to a broad
maximum at α ≈ 0.4–0.6 (B=96; 0.6–0.7 at B=128), then falls steeply: below LRU at α ≥ 0.9 for the raw score and at
α = 0.99 for z. The three signals (raw score, z relative to the K-th selected, all-page percentile) have nearly the same curve; z
is the most robust at high α. So **some memory beats none, but too much memory is worse than LRU**. The best α grows with
the budget (frozen held-out values: 0.4 → 0.95 from B=72 to 256).

Source: `results_v2/dev/s4/alpha_curve/alpha_curve.csv` (every signal × α × B) and `alpha_curve.png` (B = 96, 128, 256,
with the current score and InfLLM Eq. 1 as reference lines).

| α | raw @96 | z @96 | allrank @96 | z @128 |
|---|---|---|---|---|
| 0 (= current score) | 0.240 | 0.240 | 0.240 | 0.216 |
| 0.2 | 0.260 | 0.261 | 0.253 | 0.236 |
| 0.4 | **0.270** | 0.275 | 0.265 | 0.254 |
| 0.5 | 0.265 | **0.279** | 0.270 | 0.261 |
| 0.6 | 0.247 | 0.277 | **0.271** | 0.267 |
| 0.7 | 0.201 | 0.267 | 0.268 | **0.268** |
| 0.8 | 0.101 | 0.248 | 0.256 | 0.259 |
| 0.9 | −0.147 | 0.195 | 0.227 | 0.222 |
| 0.95 | −0.446 | 0.088 | 0.180 | 0.170 |
| 0.99 | −1.114 | −0.393 | −0.146 | −0.160 |

For the paper: draw this as one curve with ArkVale/FlashOffloading at the left end and frequency-style policies (decayed
count, LFU: G −0.86 at B=96) toward the right, instead of as unrelated competitors.

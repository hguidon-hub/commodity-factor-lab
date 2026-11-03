## Question
Does **trader positioning**, as reported weekly by the CFTC, help machine learning rank commodity futures by next-month return, beyond what price-based signals already capture?

## Data
| Block | Source | Frequency | Notes |
|---|---|---|---|
| Prices | Yahoo Finance continuous front-month futures (`CL=F`, `GC=F`, …) | daily → month-end | Large roll jumps neutralised with a robust MAD filter |
| Positioning (legacy) | CFTC Commitments of Traders, futures only: commercials vs non-commercials | weekly (Tuesday) | Lagged to the Friday release, +3 days: no look-ahead |
| Positioning (disaggregated) | CFTC: producers/merchants, swap dealers, managed money, top-4 concentration | weekly, 2006+ | Same lag |

Universe: 26 liquid futures across energy, metals, grains, softs and livestock. Brent (ICE) has no CFTC data.

## Features (known at month-end *t*)
- **Price (7):** reversal (1m), momentum (3m, 6m, 12-1m), realised volatility, skewness, max daily return.
- **Positioning (up to 14):**
  - commercials' hedging pressure (level and 3-year z-score);
  - non-commercials' net position and its change;
  - change in open interest;
  - managed-money net position (level, z-score, 1-month change, crowding = percentile of its own 3-year range);
  - producer and swap-dealer net positions;
  - top-4 traders' concentration.

All features are **uniformised cross-sectionally** (rank → [0, 1]), following Coqueret & Guida (2020). Missing values are set to the neutral 0.5. The current month is dropped until it is complete.

## Models and the ablation
Ridge regression, Random Forest, Gradient Boosting (histogram), MLP, and an **ensemble** that averages each model's cross-sectional rank. The label is the uniformised return of month *t+1*.

**Ablation:** Ridge and GBM are re-trained on price features only and on positioning features only. If positioning carries information, the full model should beat the price-only model out-of-sample.

## Validation
- **Expanding-window walk-forward**: 60 months of initial training, refit every 12 months, trained only on labels fully realised before the prediction date.
- Portfolios: long the top bucket, short the bottom bucket, with inverse-volatility weights, 100% gross per leg and transaction costs on turnover. An optional 10% ex-ante volatility target uses trailing 36-month vol, capped at 3× leverage.
- Benchmarks: 12-1 momentum, hedging pressure (long when hedgers are net short), managed-money contrarian (fade crowded speculative longs), and an equal-weight long-only basket.
- Diagnostics: monthly Spearman IC and its t-stat, permutation importance by feature and group, and turnover.

## Limitations
- **Returns are approximate.** Yahoo continuous contracts are not roll-adjusted, so roll yield is missing and return levels are biased; the long-only basket is the most affected. Cross-sectional comparisons are more robust.
- **No carry factor** without the second contract. It is empirically the strongest commodity premium, so its absence penalises the price-only side.
- Small cross-section (about 25 assets): ICs are noisy, and an |IC t-stat| below 2 cannot be distinguished from zero.
- Hyper-parameters were fixed a priori (no tuning) to limit overfitting.
- Transaction costs are a flat bps assumption.

## References
- Coqueret, G. & Guida, T. (2020). *Machine Learning for Factor Investing*. CRC Press.
- Basu, D. & Miffre, J. (2013). Capturing the risk premium of commodity futures: the role of hedging pressure. *JBF*.
- Gorton, G. & Rouwenhorst, K. G. (2006). Facts and Fantasies about Commodity Futures. *FAJ*.
- Koijen, R., Moskowitz, T., Pedersen, L. & Vrugt, E. (2018). Carry. *JFE*.
- Bakshi, G., Gao, X. & Rossi, A. (2019). Understanding the sources of risk underlying the cross section of commodity returns. *Management Science*.

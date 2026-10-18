## Question
Can machine learning combine classic commodity risk premia — **momentum, carry, hedging pressure, speculative positioning, volatility and skewness** — into a better cross-sectional signal than each factor alone, once evaluated strictly out-of-sample?

## Data
| Block | Source | Frequency | Notes |
|---|---|---|---|
| Prices | Yahoo Finance continuous front-month futures (`CL=F`, `GC=F`, …) or a Bloomberg export (`CL1`, `CL2` Comdty) | daily → month-end | Roll jumps neutralised with a robust MAD filter |
| Positioning | CFTC Commitments of Traders, legacy futures-only report | weekly (Tuesday) | Lagged to Friday release + 3 days: no look-ahead |
| Carry | 2nd nearby contract (Bloomberg only) | daily | `ln(F1/F2)` — backwardation = positive carry |

Universe: 26 liquid futures across energy, metals, grains, softs and livestock.

## Features (known at month-end *t*)
Reversal (1m), momentum (3m, 6m, 12-1m), realised volatility, skewness, max daily return, carry and its change, commercials' hedging pressure (level and 3-year z-score), speculators' net position and its change, change in open interest.

All features are **uniformised cross-sectionally** (rank → [0, 1]), following Coqueret & Guida (2020). Missing values are set to the neutral 0.5.

## Models
Ridge regression, Random Forest, Gradient Boosting (histogram), MLP, and an **ensemble** that averages each model's cross-sectional rank. The label is the uniformised return of month *t+1*.

## Validation
- **Expanding-window walk-forward**: 60 months of initial training, refit every 12 months, trained only on labels fully realised before the prediction date.
- Portfolios: long the top bucket, short the bottom bucket, with inverse-volatility weights, 100% gross per leg, and transaction costs on turnover. An optional 10% ex-ante volatility target uses trailing 36-month vol, capped at 3× leverage.
- Benchmarks: single-factor sorts (12-1 momentum, hedging pressure, carry) and an equal-weight long-only basket.
- Diagnostics: monthly Spearman IC and its t-stat, permutation importance (drop in IC when a feature is shuffled within each month), and turnover.

## Limitations
- Yahoo continuous contracts are not roll-adjusted. The MAD filter removes the biggest roll jumps, but returns ignore roll yield, so a Bloomberg export is the correct source.
- Small cross-section (about 26 assets): ICs are noisy and the results are sensitive to the bucket size.
- Hyper-parameters were fixed a priori (no tuning) to limit overfitting, so the results may understate what the models can do.
- Brent (ICE) has no COT data in the CFTC legacy report.
- Transaction costs are a flat bps assumption. They ignore roll costs, margin, and market impact in thin markets (OJ, oats, rice).

## References
- Coqueret, G. & Guida, T. (2020). *Machine Learning for Factor Investing*. CRC Press.
- Gorton, G. & Rouwenhorst, K. G. (2006). Facts and Fantasies about Commodity Futures. *FAJ*.
- Koijen, R., Moskowitz, T., Pedersen, L. & Vrugt, E. (2018). Carry. *JFE*.
- Basu, D. & Miffre, J. (2013). Capturing the risk premium of commodity futures: the role of hedging pressure. *JBF*.
- Bakshi, G., Gao, X. & Rossi, A. (2019). Understanding the sources of risk underlying the cross section of commodity returns. *Management Science*.

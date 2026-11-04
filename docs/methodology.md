## Question
Does **trader positioning**, as reported weekly by the CFTC, help machine learning rank commodities by next-month return, beyond what price-based signals already capture?

## Data
| Block | Source | Frequency | Notes |
|---|---|---|---|
| Prices | 13 single-commodity ETFs on Yahoo Finance (USO, UNG, GLD, CORN, …), dividend/split-adjusted | daily → month-end | Investable returns: the futures roll (or metal storage) is embedded |
| Positioning (legacy) | CFTC Commitments of Traders, futures only: commercials vs non-commercials | weekly (Tuesday) | Lagged to the Friday release, +3 days: no look-ahead |
| Positioning (disaggregated) | CFTC: producers/merchants, swap dealers, managed money, top-4 concentration | weekly, 2006+ | Same lag |

Universe: 13 ETFs across energy, metals, grains and softs, each mapped to the CFTC report of its underlying futures contract. The market names actually matched are saved with every run and shown in the app, so a wrong contract code is visible. Brent (BNO, ICE) has no CFTC data; its positioning features are set to neutral. The cross-section needs at least 8 ETFs, so the panel starts in 2010, and the first out-of-sample month comes after 48 months of training.

## Features (known at month-end *t*)
- **Price (7):** reversal (1m), momentum (3m, 6m, 12-1m), realised volatility, skewness, max daily return.
- **Positioning (up to 17):**
  - commercials' hedging pressure (level, 3-year z-score and 1-month change);
  - non-commercials' net position and its change;
  - change in open interest;
  - managed-money net position (level, z-score, 1-month change, crowding = percentile of its own 3-year range);
  - producer and swap-dealer net positions, and their 1-month changes (flows);
  - top-4 traders' concentration.

All features are **uniformised cross-sectionally** (rank → [0, 1]), following Coqueret & Guida (2020). Missing values are set to the neutral 0.5. The current month is dropped until it is complete.

## Models and the ablation
Ridge regression, Random Forest, Gradient Boosting (histogram), MLP, and an **ensemble** that averages each model's cross-sectional rank. The label is the uniformised return of month *t+1*.

**Ablation:** Ridge and GBM are re-trained on price features only and on positioning features only. If positioning carries information, the full model should beat the price-only model out-of-sample.

**Feature selection:** a further Ridge keeps, at each yearly refit, the 6 features with the highest |IC t-stat| on the training window only. This tests whether fewer, better-chosen inputs beat all 24, without any look-ahead (a unit test checks that the selection uses past data only). k = 6 is fixed in advance; k = 4, 8 and 10 are run as a sensitivity check. Limit: each signal is scored alone, so near-duplicates (managed money and non-commercials) can both be kept.

## Validation
- **Expanding-window walk-forward**: 48 months of initial training, refit every 12 months, trained only on labels fully realised before the prediction date.
- Portfolios: long the top bucket, short the bottom bucket, with inverse-volatility weights, 100% gross per leg and transaction costs on turnover. An optional 10% ex-ante volatility target uses trailing 36-month vol, capped at 3× leverage.
- Benchmarks: 12-1 momentum, hedging pressure (long when hedgers are net short), managed-money contrarian (fade crowded speculative longs), and an equal-weight long-only basket.
- Diagnostics: monthly Spearman IC and its t-stat, permutation importance by feature and group, and turnover.
- Robustness: results split before / after 2020, transaction costs from 0 to 30 bps, quartile / tercile / half portfolios, and a multiple-testing check (with ~12 signals, a Bonferroni-style bar is t ≈ 2.9).

## Limitations
- **Small universe and short history.** 13 ETFs from 2010 onward: ICs are noisy, and an |IC t-stat| below 2 cannot be distinguished from zero.
- **ETF frictions.** Management fees, tracking error and some structural changes (e.g. USO changing its roll policy in 2020) are embedded in the returns. No livestock, coffee, cocoa or cotton ETF has a long enough clean history.
- **No carry factor** without second-contract prices. It is empirically the strongest commodity premium, so its absence penalises the price-only side.
- Hyper-parameters were fixed a priori (no tuning) to limit overfitting.
- Transaction costs are a flat bps assumption.

## References
- Coqueret, G. & Guida, T. (2020). *Machine Learning for Factor Investing*. CRC Press.
- Basu, D. & Miffre, J. (2013). Capturing the risk premium of commodity futures: the role of hedging pressure. *JBF*.
- Gorton, G. & Rouwenhorst, K. G. (2006). Facts and Fantasies about Commodity Futures. *FAJ*.
- Koijen, R., Moskowitz, T., Pedersen, L. & Vrugt, E. (2018). Carry. *JFE*.
- Bakshi, G., Gao, X. & Rossi, A. (2019). Understanding the sources of risk underlying the cross section of commodity returns. *Management Science*.

# 🛢️ Commodity Factor Lab

**Does trader positioning help machine learning predict commodity returns?** This project combines price signals with the CFTC Commitments of Traders (who holds what: producers, swap dealers, hedge funds) for 13 single-commodity ETFs. Every model is tested strictly out-of-sample, and the dashboard updates itself every month.

> emlyon MSc in Finance — *Machine Learning for Finance* (G. Coqueret), 2026–27 project.
> **🔗 Live app: [hguidon-hub-hguidon-commodity-lab.share.connect.posit.cloud](https://hguidon-hub-hguidon-commodity-lab.share.connect.posit.cloud)** · Author: Hugo Guidon

## Research question
Price-based commodity factors (momentum, reversal, volatility, skewness) are well documented. Positioning data is specific to futures markets and much less used in ML studies. The core test is an **ablation**: the same models are trained on
1. price features only,
2. positioning features only,
3. both,

and the out-of-sample rank IC and long/short Sharpe ratios are compared.

## What it does
1. **Data**: daily adjusted prices of 13 single-commodity ETFs (USO, BNO, UNG, UGA, GLD, SLV, CPER, PPLT, PALL, CORN, WEAT, SOYB, CANE) from Yahoo Finance, plus two weekly **CFTC Commitments of Traders** reports (legacy and disaggregated, 2006 onward), all free and public.
2. **Features**: 7 price features and up to 17 positioning features (levels and 1-month flows): hedging pressure, managed-money net position and crowding, producer and swap-dealer positioning, top-4 concentration. All are uniformised cross-sectionally.
3. **Models**: Ridge, Random Forest, Gradient Boosting, MLP and a rank ensemble, trained walk-forward on an expanding window and refit every 12 months. Ridge and GBM are also trained on each feature group alone, for the ablation.
4. **Portfolio**: long/short bucket sorts with inverse-vol weights, transaction costs and a 10% vol target, compared against single-factor benchmarks (momentum, hedging pressure, managed-money contrarian) and an equal-weight long-only basket.
5. **App**: a plain-language "Start here" page with an auto-updated verdict, the ablation ("Does positioning help?"), performance, **robustness** (sub-periods, costs, portfolio rules, multiple-testing warning), the current month's book, ICs, feature importance by group, a factor explorer and the methodology.
6. **Self-updating**: a GitHub Action runs on the 3rd of each month and after every code change. It re-downloads the data, re-runs the models and commits the results, and Posit Connect Cloud then republishes the app.

## Why ETFs?
ETF prices embed the cost of rolling the underlying futures (or of holding the metal), so their returns are what an investor actually earned. Continuous front-month futures series from free sources ignore the roll and distort return levels. The trade-off is a smaller cross-section (13 assets) and a history that starts around 2010. Each ETF is mapped to the CFTC report of its underlying contract. A futures universe and a Bloomberg/Refinitiv loader with 1st and 2nd contracts (which unlocks the carry factor) remain in `src/cfl/` as an extension.

## Quick start
```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python scripts/run_pipeline.py --source synthetic --out data/results_demo   # offline demo
python scripts/run_pipeline.py --source yahoo                               # live data -> data/results
streamlit run app/streamlit_app.py
pytest -q
```

## Deploy (free)
The app is hosted on [Posit Connect Cloud](https://connect.posit.cloud): *Publish* → *Streamlit* → this repo, branch `main`, primary file `app/streamlit_app.py`, with *Automatically publish on push* enabled.

## Structure
```
src/cfl/        config (universe) · data (Yahoo / CFTC legacy + disaggregated / synthetic) · features · models · backtest
scripts/        run_pipeline.py — end-to-end
app/            streamlit_app.py — dashboard
tests/          synthetic-data tests (no network, look-ahead and ablation checks)
docs/           methodology · report outline · slides outline · LLM usage log
.github/        monthly + on-push self-update workflow
```

*Educational project, not investment advice.*

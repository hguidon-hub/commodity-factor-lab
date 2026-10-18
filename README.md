# 🛢️ Commodity Factor Lab

**Machine learning for the cross-section of commodity futures returns**: momentum, carry, hedging pressure and CFTC positioning, combined by ML models and evaluated strictly out-of-sample. The dashboard updates itself every month.

> emlyon MSc in Finance — *Machine Learning for Factor Investing* (G. Coqueret), 2026–27 project.
> Live app: `https://<your-app>.streamlit.app` · Author: Hugo Guidon

## What it does
1. **Data**: 26 futures (energy, metals, grains, softs, livestock) from Yahoo Finance or a Bloomberg export, plus the weekly **CFTC Commitments of Traders** report.
2. **Features**: 14 monthly characteristics (reversal, momentum, vol, skew, lottery, carry, hedging pressure, speculative net position, open interest), uniformised cross-sectionally.
3. **Models**: Ridge, Random Forest, Gradient Boosting, MLP and a rank ensemble, trained walk-forward on an expanding window and refit every 12 months.
4. **Portfolio**: long/short bucket sorts with inverse-vol weights, transaction costs and a 10% vol target, compared against single-factor benchmarks.
5. **App**: performance, the current month's long/short book, ICs, feature importance, factor explorer and methodology.
6. **Self-updating**: a GitHub Action runs on the 3rd of each month, re-downloads the data, re-runs the models and commits the results. Streamlit Cloud then redeploys.

## Quick start
```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python scripts/run_pipeline.py --source synthetic --out data/results_demo   # offline demo (~4 min)
python scripts/run_pipeline.py --source yahoo                               # live data -> data/results
streamlit run app/streamlit_app.py
pytest -q
```

### Using Bloomberg (recommended, emlyon trading room)
Use `BDH` with `PX_SETTLE` on generic 1st **and** 2nd contracts (`CL1 Comdty`, `CL2 Comdty`, `GC1`, `GC2`…, roots in `src/cfl/config.py`) from 2006 onward. Export to CSV with dates in the first column, then run:
```bash
python scripts/run_pipeline.py --source bloomberg --bbg data/raw/bbg_px_settle.csv
```
This gives proper roll handling and unlocks the **carry** factor.

## Deploy (free)
1. Push this folder to a **public GitHub repo**.
2. Go to [share.streamlit.io](https://share.streamlit.io), click *New app*, pick the repo, and set the main file to `app/streamlit_app.py`.
3. In GitHub, open *Actions* → *Monthly data & model update* → *Run workflow* once. After that it runs every month.

## Structure
```
src/cfl/        config (universe) · data (Yahoo/Bloomberg/CFTC/synthetic) · features · models · backtest
scripts/        run_pipeline.py — end-to-end
app/            streamlit_app.py — dashboard
tests/          synthetic-data tests (no network, look-ahead check)
docs/           methodology · report outline · slides outline · LLM usage log
.github/        monthly self-update workflow
```

*Educational project, not investment advice.*

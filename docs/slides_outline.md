# Oral — session 7, 5 minutes, 3 slides

**Slide 1 — The question (1 min)**
"Can ML beat the classic commodity factors?" Show the universe (26 futures, 5 sectors) and the 3 known premia: momentum, carry, hedging pressure. Close with why it matters: this is how commodity trading desks think about positioning.

**Slide 2 — The machine (2 min)**
One pipeline diagram: Yahoo/Bloomberg + CFTC COT → 14 features → walk-forward ML (Ridge, RF, GBM, MLP, ensemble) → long/short book → self-updating Streamlit app. Keep one bullet on avoiding look-ahead (COT release lag, expanding window).

**Slide 3 — First results & roadmap (2 min)**
Show the equity curve of the ensemble vs single factors, plus the OOS Sharpe/IC table and the top 3 features. Then the roadmap to December: Bloomberg carry, robustness tests, SHAP, macro regimes. Live-demo the URL if possible.

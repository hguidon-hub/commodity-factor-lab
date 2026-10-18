# Oral — session 7 (intermediate checkpoint): 5 minutes, 3 slides

Goal: pitch the idea and the planned structure. No final results expected.

**Slide 1 — Idea & motivation (1 min 30)**
- Question: can ML combine known commodity risk premia (momentum, carry, hedging pressure, speculative positioning) into a better cross-sectional signal than each factor alone?
- Why it is original: most projects stay on equities and Fama-French factors. Commodities add a dimension specific to futures: the CFTC positioning data (who holds what).
- Why it matters: this is how commodity trading houses think about positioning (career link).

**Slide 2 — Planned structure / pipeline (2 min)**
One left-to-right diagram:
Data (Bloomberg generic 1st/2nd contracts + CFTC COT, 26 futures, 2006→) → 14 features, ranked cross-sectionally → walk-forward ML (Ridge, RF, GBM, MLP, ensemble) → long/short portfolio with costs & vol target → self-updating Streamlit app (monthly GitHub Action).
One bullet on rigour: no look-ahead (COT release lag, expanding window).

**Slide 3 — Status & roadmap to 31 Dec (1 min 30)**
- Done: pipeline, app skeleton, automated monthly update (show the URL / a screenshot).
- First lesson: free Yahoo data gives no exploitable signal (front-month prices ≠ futures returns, no carry), so the project is switching to Bloomberg.
- Next: Bloomberg data and carry → robustness tests (costs, buckets, sub-periods) → explainability (feature importance / SHAP) → possibly macro regimes → report.
- Open question for the professor: frequency (monthly vs weekly), and whether the COT should also be used as a timing signal.

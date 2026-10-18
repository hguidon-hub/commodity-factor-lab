# Written report — outline (2–5 pages)

Specs checklist: process of creation · sample prompts · generative tools used · problems & solutions · limitations · extensions.

1. **Motivation (½ p)** — Commodity premia are well documented one factor at a time: momentum, carry (Koijen et al.) and hedging pressure (Basu & Miffre). Can ML combine them and add value out-of-sample? Why it matters for a commodity trading house (LDC, Mercuria, Trafigura…).
2. **Object (½ p)** — URL, screenshots of the app, what a user can do (change the bucket size, transaction costs and vol target, read the current book).
3. **Data & features (¾ p)** — Sources and timing table (COT release lag!), roll-jump cleaning, uniformisation.
4. **Models & validation (¾ p)** — Walk-forward design and why random K-fold would leak. Model zoo, ensemble, benchmarks.
5. **Results (1 p)** — OOS table (Sharpe, IC, t-stat, turnover), equity curves, feature importance, robustness checks (bucket size, costs, sub-periods 2011–15 vs 2020–22).
6. **Process & generative AI (½ p)** — Tools, how they were used, sample prompts (see `llm_log.md`), what you verified and rewrote yourself.
7. **Problems & solutions (½ p)** — e.g. Yahoo roll jumps → MAD filter; COT contract codes changing over time; Brent missing from CFTC; small cross-section → noisy IC; free hosting limits.
8. **Limitations & extensions (½ p)** — Roll-adjusted data, term-structure features (F1–F12), macro regime conditioning (DXY, real rates, PMI), weekly frequency, SHAP, hyper-parameter tuning with purged CV, LLM sentiment from USDA/EIA reports.

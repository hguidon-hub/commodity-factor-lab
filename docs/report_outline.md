# Written report — outline (2–5 pages)

Specs checklist: process of creation · sample prompts · generative tools used · problems & solutions · limitations · extensions.

1. **Motivation (½ p)** — Positioning data is specific to futures markets. Does it add information beyond price factors? Why it matters for commodity trading houses.
2. **Object (½ p)** — The app URL and screenshots of the ablation tab, the performance tab and the current book. What a user can do with it.
3. **Data & features (¾ p)** — Yahoo prices plus the legacy and disaggregated CFTC reports. Data timing (release lag), roll-jump cleaning, uniformisation, and the two feature groups.
4. **Models & validation (¾ p)** — Walk-forward design (why random K-fold would leak), the model zoo, the ensemble, the ablation design and the benchmarks.
5. **Results (1 p)** — The ablation table (IC, IC t-stat and Sharpe for price only, positioning only and both), feature importance by group, the managed-money contrarian benchmark, and robustness to costs, bucket size and sub-periods.
6. **Process & generative AI (½ p)** — Tools, sample prompts (see `llm_log.md`), and what was verified and rewritten by hand.
7. **Problems & solutions (½ p)** — No access to second-contract data on the school Bloomberg: return levels approximated with front-month prices, carry left as an extension. Also: Yahoo roll jumps (MAD filter), COT release timing, Brent missing from the CFTC.
8. **Limitations & extensions (½ p)** — Roll-adjusted data and carry, weekly frequency (COT is weekly), macro regimes, SHAP, tuning with purged CV, LLM sentiment from USDA/EIA reports.

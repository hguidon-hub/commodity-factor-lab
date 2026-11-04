# Oral — session 7 (intermediate checkpoint): 5 minutes, 3 slides

Goal: pitch the idea and the planned structure. No final results expected.

**Slide 1 — Idea (1 min 30)**
Can machine learning tell which commodities will outperform next month, and does knowing **who holds what** (CFTC positioning) help? Commodities are less studied than stocks and close to how trading desks think.

**Slide 2 — Planned structure (2 min)**
Data (13 commodity ETFs + CFTC positions) → factors (price vs positioning) → ML models tested out-of-sample → long/short portfolio and a live app. The key experiment: the same models with and without positioning.

**Slide 3 — Status & roadmap (1 min 30)**
- Done: pipeline and online app updated monthly; faithful data (ETF returns + detailed CFTC report); first test with vs without positioning.
- Next, by 31 Dec: robustness checks, sharper signals (feature selection, weekly data), then the final app and report.

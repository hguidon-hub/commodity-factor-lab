# Generative AI usage log (Rule 1 — mention when/how LLMs are used)

Keep this up to date as you go. It feeds section 6 of the report.

| Date | Tool | What for | Sample prompt | What I checked / changed myself |
|---|---|---|---|---|
| 2026-10-18 | Claude (Anthropic) | Brainstorm project ideas from the specs; scaffold the repo (data loaders, features, walk-forward backtest, Streamlit app, CI workflow) | "Read these project specs, propose ideas and create the project" | Picked the commodity topic myself (career fit). To do: verify CFTC contract codes, run on live data, review every feature definition against the literature |
| | | | | |

## Things to verify personally (do not trust the scaffold blindly)
- [ ] CFTC contract market codes in `config.py` map to the right contracts across 2006–2026
- [ ] Hedging pressure sign convention vs Basu & Miffre (2013)
- [ ] Roll-jump filter threshold (z = 6): inspect detected jumps for CL, NG, LE
- [ ] Walk-forward has no look-ahead (test included; re-read `walk_forward`)
- [ ] Results are robust to the bucket size, costs and sub-periods

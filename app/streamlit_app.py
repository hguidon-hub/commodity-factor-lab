"""Commodity Factor Lab — Streamlit dashboard.

Run locally:  streamlit run app/streamlit_app.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from cfl import backtest as bt                  # noqa: E402
from cfl.config import SECTOR, UNIVERSE         # noqa: E402
from cfl.features import FEATURE_DOC, feature_group   # noqa: E402

st.set_page_config(page_title="Commodity Factor Lab", page_icon="🛢️", layout="wide")

# ------------------------------------------------------------------ look & feel
INK, MUTED, PAPER, CARD, LINE = "#1F2933", "#52606D", "#FFFDF8", "#FFFFFF", "#E6E0D4"
PALETTE = ["#E8590C", "#1C7ED6", "#2F9E44", "#AE3EC9", "#F59F00", "#0CA678", "#D6336C", "#5C7CFA", "#868E96"]
SECTOR_COLORS = {"Energy": "#E8590C", "Metals": "#1C7ED6", "Grains": "#F59F00", "Softs": "#2F9E44",
                 "Livestock": "#AE3EC9"}
SET_COLORS = {"Price only": "#1C7ED6", "Positioning only": "#E8590C", "Price + positioning": "#2F9E44"}

pio.templates["cfl"] = go.layout.Template(layout=dict(
    font=dict(family="Inter, -apple-system, Segoe UI, sans-serif", color=INK, size=13),
    paper_bgcolor=CARD, plot_bgcolor=CARD, colorway=PALETTE,
    xaxis=dict(gridcolor="#F1ECE2", linecolor=LINE, zerolinecolor=LINE),
    yaxis=dict(gridcolor="#F1ECE2", linecolor=LINE, zerolinecolor="#CBC3B3"),
    legend=dict(bgcolor="rgba(0,0,0,0)"), margin=dict(t=40, b=20, l=10, r=10),
    hoverlabel=dict(bgcolor="#FFFFFF", font_color=INK)))
pio.templates.default = "plotly_white+cfl"

st.markdown(f"""
<style>
.block-container {{padding-top: 2rem; max-width: 1400px;}}
.hero {{background: linear-gradient(120deg, #FFF1E6 0%, #E7F5FF 55%, #EBFBEE 100%);
        border: 1px solid {LINE}; border-radius: 18px; padding: 28px 32px; margin-bottom: 18px;}}
.hero h1 {{font-size: 2.3rem; margin: 0 0 6px 0; color: {INK};}}
.hero p {{font-size: 1.08rem; color: {MUTED}; margin: 0; max-width: 950px;}}
.pill {{display: inline-block; padding: 4px 12px; border-radius: 999px; font-size: .85rem; font-weight: 600;
        margin: 12px 8px 0 0;}}
.card {{background: {CARD}; border: 1px solid {LINE}; border-radius: 14px; padding: 18px 20px; height: 100%;}}
.card h4 {{margin: 0 0 6px 0; font-size: 1.02rem;}}
.card p {{margin: 0; color: {MUTED}; font-size: .95rem;}}
.step {{font-weight: 700; font-size: .8rem; letter-spacing: .08em;}}
.verdict {{border-left: 6px solid #E8590C; background: #FFF4E6; border-radius: 10px; padding: 16px 20px;
           font-size: 1.05rem; color: {INK};}}
div[data-testid="stMetric"] {{background: {CARD}; border: 1px solid {LINE}; border-radius: 14px; padding: 12px 16px;}}
</style>""", unsafe_allow_html=True)


def card(title: str, text: str, color: str, step: str | None = None) -> str:
    s = f'<div class="step" style="color:{color}">{step}</div>' if step else ""
    return f'<div class="card" style="border-top:5px solid {color}">{s}<h4>{title}</h4><p>{text}</p></div>'


def how_to_read(text: str) -> None:
    with st.expander("ℹ️ How to read this"):
        st.markdown(text)


# ------------------------------------------------------------------ data
LIVE, DEMO = ROOT / "data" / "results", ROOT / "data" / "results_demo"
RES = LIVE if (LIVE / "meta.json").exists() else DEMO
NAMES = {c.yf: c.name for c in UNIVERSE}
ML = ["Ridge", "RandomForest", "GBM", "MLP", "Ensemble"]


@st.cache_data
def load(res: str):
    r = Path(res)
    meta = json.loads((r / "meta.json").read_text())
    P = pd.read_csv(r / "predictions.csv.gz", index_col=[0, 1], parse_dates=[0])
    panel = pd.read_csv(r / "panel.csv.gz", index_col=[0, 1], parse_dates=[0])
    live = pd.read_csv(r / "live_signals.csv", index_col=0)
    imp = pd.read_csv(r / "importance.csv", index_col=0).iloc[:, 0]
    ic = pd.read_csv(r / "ic.csv", index_col=0, parse_dates=True)
    return meta, P, panel, live, imp, ic


@st.cache_data
def run_portfolios(res: str, q: float, tc: float, inv_vol: bool, target: float | None):
    _, P, panel, *_ = load(res)
    strategies = [c for c in P.columns if c != "fwd_ret"]
    rets, turns = bt.portfolio_returns(P, strategies, q=q, tc_bps=tc,
                                       inv_vol=panel["raw_vol_3m"] if inv_vol else None)
    if target:
        rets = bt.vol_target(rets, target=target)
    return rets, turns


meta, P, panel, live, imp, ic = load(str(RES))
strategies_all = [c for c in P.columns if c != "fwd_ret"] + ["EW long-only"]

# ------------------------------------------------------------------ sidebar
with st.sidebar:
    st.markdown("### 🛢️ Commodity Factor Lab")
    st.caption("Does trader positioning help machine learning predict commodity returns?")
    if RES == DEMO:
        st.warning("Running on **synthetic demo data**.")
    st.markdown(f"**Signal date:** {meta['signal_date']}  \n**Last update:** {meta['run_utc'][:10]}  \n"
                f"**Universe:** {meta['n_assets']} commodity ETFs  \n**Data:** ETF prices (Yahoo) + CFTC COT")
    st.divider()
    st.markdown("#### ⚙️ Portfolio settings")
    st.caption("These settings change every portfolio on the site, live.")
    q = st.select_slider("Long/short bucket", options=[0.2, 0.25, 1 / 3, 0.5], value=1 / 3,
                         format_func=lambda x: {0.2: "Quintiles", 0.25: "Quartiles", 1/3: "Terciles", 0.5: "Halves"}[x],
                         help="Terciles = buy the best third of commodities, sell the worst third.")
    tc = st.slider("Transaction cost (bps per unit turnover)", 0, 30, 5,
                   help="Cost charged each month on the traded amount. 5 bps = 0.05%.")
    inv_vol = st.toggle("Inverse-volatility weights", value=True,
                        help="Calmer commodities get a bigger weight, so that natural gas does not dominate the book.")
    vt = st.toggle("Target 10% volatility", value=True,
                   help="Scales every strategy to the same risk level, so their returns are comparable.")
    st.divider()
    st.caption("Built by Hugo Guidon — emlyon MSc in Finance, *Machine Learning for Finance* "
               "(Prof. G. Coqueret). Educational project, not investment advice.")

rets, turns = run_portfolios(str(RES), q, tc, inv_vol, 0.10 if vt else None)
table = bt.summary(rets, turns, ic)


def ablation_rows() -> pd.DataFrame:
    rows = []
    for m in ["Ridge", "GBM"]:
        for label, name in [("Price only", f"{m} [price only]"), ("Positioning only", f"{m} [positioning only]"),
                            ("Price + positioning", m)]:
            if name in table.index:
                rows.append({"Model": m, "Information set": label, "Mean IC": table.loc[name, "Mean IC"],
                             "IC t-stat": table.loc[name, "IC t-stat"], "Sharpe": table.loc[name, "Sharpe"]})
    return pd.DataFrame(rows)


abl = ablation_rows()
ml_t = table.loc[[c for c in table.index if c in ML or "[" in c], "IC t-stat"].abs()
years_oos = (pd.Timestamp(meta["signal_date"]) - pd.Timestamp(meta["first_oos"])).days / 365.25

# ------------------------------------------------------------------ hero
st.markdown(f"""
<div class="hero">
  <h1>🛢️ Commodity Factor Lab</h1>
  <p>Every week, the US regulator (CFTC) publishes <b>who holds what</b> in commodity futures: producers, banks,
  hedge funds. This project tests whether that information, combined with classic price signals, lets
  machine-learning models predict which commodities will do best <b>next month</b>, using only data
  that was available at the time.</p>
  <span class="pill" style="background:#FFE8CC;color:#B34700">{meta['n_assets']} commodity ETFs</span>
  <span class="pill" style="background:#D0EBFF;color:#1450A3">{len(meta['features'])} signals</span>
  <span class="pill" style="background:#D3F9D8;color:#1E6B30">{years_oos:.0f} years tested out-of-sample</span>
  <span class="pill" style="background:#F3D9FA;color:#7B2A8F">updated every month</span>
</div>""", unsafe_allow_html=True)

tab_home, tab_pos, tab_perf, tab_rob, tab_live, tab_diag, tab_fact, tab_method = st.tabs(
    ["🏠 Start here", "🧪 Does positioning help?", "📈 Performance", "🛡️ Robustness", "🎯 This month's book",
     "🔬 Model diagnostics", "🧭 Factor explorer", "📚 Methodology"])

# ------------------------------------------------------------------ start here
with tab_home:
    st.markdown("### The idea in four steps")
    cols = st.columns(4)
    steps = [
        ("Collect the data", f"Daily prices of {meta['n_assets']} single-commodity ETFs (oil, gas, gold, copper, "
         "wheat, sugar…), which track real futures returns, and the weekly CFTC report on traders' positions.",
         PALETTE[0]),
        ("Build signals", "Price signals (trend, reversal, volatility) and positioning signals (are hedge funds "
         "crowded? are producers hedging heavily?), ranked from 0 to 1 each month.", PALETTE[1]),
        ("Train ML models", "Linear model, random forest, gradient boosting and neural network. They are retrained every "
         "year and only ever predict months they have not seen.", PALETTE[2]),
        ("Build a portfolio", "Each month: buy the commodities with the best predicted ranks, sell the worst, "
         "pay transaction costs, and track the result.", PALETTE[3]),
    ]
    for c, (t, txt, col), i in zip(cols, steps, range(1, 5)):
        c.markdown(card(t, txt, col, f"STEP {i}"), unsafe_allow_html=True)

    st.markdown("### What the results say so far")
    sig = table.dropna(subset=["IC t-stat"])
    n_tests = len(sig)
    t_bonf = 2.9 if n_tests >= 10 else 2.6        # ~5% family-wise threshold for this many tests
    passing = sig[sig["IC t-stat"] >= 2].sort_values("IC t-stat", ascending=False)
    if passing.empty:
        verdict = ("<b>No signal beats chance with statistical confidence</b> (every IC t-stat is below 2). "
                   "On this data, neither prices nor positioning give a reliable edge for a monthly ranking strategy.")
    else:
        names = ", ".join(f"<b>{i}</b> (t = {r['IC t-stat']:.1f})" for i, r in passing.iterrows())
        verdict = (f"<b>{len(passing)} of {n_tests} signals clear the usual t > 2 bar:</b> {names}. ")
        verdict += ("At least one of them also survives a strict multiple-testing threshold (t > "
                    f"{t_bonf}).") if (passing["IC t-stat"] >= t_bonf).any() else (
                    f"None survives a strict multiple-testing threshold (t > {t_bonf}): with {n_tests} signals "
                    "tested, one or two crossing t = 2 by luck is plausible, so these are encouraging, not proven.")
    try:
        r = abl[abl["Model"] == "Ridge"].set_index("Information set")["Mean IC"]
        delta = r["Price + positioning"] - r["Price only"]
        verdict += (f"<br><br><b>Does positioning help?</b> For the linear model, mean IC is {r['Price only']:+.3f} with "
                    f"prices only, {r['Positioning only']:+.3f} with positioning only and {r['Price + positioning']:+.3f} "
                    "with both: positioning "
                    + ("adds information on top of prices." if delta > 0.005 else
                       "carries signal on its own but overlaps with prices rather than adding to them." if r["Positioning only"] > 0.02
                       else "adds little or nothing.")
                    )
    except KeyError:
        pass
    if "EW long-only" in table.index:
        verdict += (f"<br><br>Reference: buying every commodity in equal weight earned a Sharpe of "
                    f"{table.loc['EW long-only', 'Sharpe']:.2f} over the same period.")
    st.markdown(f'<div class="verdict">{verdict}</div>', unsafe_allow_html=True)

    st.markdown("### Key numbers")
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Out-of-sample period", f"{meta['first_oos'][:4]}–{meta['signal_date'][:4]}")
    k2.metric("Best ML Sharpe (net)", f"{table.loc[[m for m in ML if m in table.index], 'Sharpe'].max():.2f}")
    k3.metric("Best single-factor Sharpe",
              f"{table.loc[[b for b in table.index if b not in ML and '[' not in b and b != 'EW long-only'], 'Sharpe'].max():.2f}")
    k4.metric("Long-only basket Sharpe", f"{table.loc['EW long-only', 'Sharpe']:.2f}",
              help="Buys every ETF in equal weight: the market reference, not a model.")

    st.markdown("### Glossary")
    g1, g2 = st.columns(2)
    g1.markdown("""
- **Long / short**: buy (bet it goes up) / sell (bet it goes down).
- **Cross-sectional**: we rank commodities *against each other* each month, not against cash.
- **Out-of-sample (OOS)**: results on months the model never saw during training.
- **Walk-forward**: train on the past, predict the next months, move forward, retrain.
- **Ablation**: removing one block of inputs to measure what it contributes.
""")
    g2.markdown("""
- **IC (information coefficient)**: rank correlation between predictions and next-month returns. 0 = no skill; 0.05 is already good for monthly finance data.
- **t-stat**: how many standard errors from zero. Below 2 ≈ could be luck.
- **Sharpe ratio**: return per unit of risk (annualised). Above 0.5 is decent, above 1 is strong.
- **COT report**: CFTC *Commitments of Traders*: positions of producers, swap dealers (banks), managed money (hedge funds).
""")

# ------------------------------------------------------------------ research question
with tab_pos:
    st.markdown("### Same models, three information sets")
    st.markdown("Each model is trained three times: with **price signals only**, with **positioning signals only**, "
                "and with **both**. If knowing who holds what adds information, the green bar should beat the blue one.")
    if abl.empty:
        st.info("Ablation results not available in this run.")
    else:
        c1, c2 = st.columns(2)
        order = {"Information set": list(SET_COLORS)}
        c1.plotly_chart(px.bar(abl, x="Model", y="Mean IC", color="Information set", barmode="group",
                               category_orders=order, color_discrete_map=SET_COLORS,
                               title="Prediction skill: mean out-of-sample rank IC")
                        .add_hline(y=0, line_color=INK, line_width=1)
                        .update_layout(height=400, legend=dict(orientation="h", y=-0.18, title="")),
                        width="stretch")
        c2.plotly_chart(px.bar(abl, x="Model", y="Sharpe", color="Information set", barmode="group",
                               category_orders=order, color_discrete_map=SET_COLORS,
                               title="Trading result: long/short Sharpe ratio, net of costs")
                        .add_hline(y=0, line_color=INK, line_width=1)
                        .update_layout(height=400, legend=dict(orientation="h", y=-0.18, title="")),
                        width="stretch")
        st.dataframe(abl.set_index(["Model", "Information set"]).style.format("{:.3f}")
                     .background_gradient(subset=["IC t-stat"], cmap="RdYlGn", vmin=-3, vmax=3),
                     width="stretch")
    how_to_read("""
- **Mean IC** measures pure prediction skill, before any trading. Bars above zero mean the model ranks
  commodities better than chance on average.
- **IC t-stat** says whether that skill is distinguishable from luck: |t| above 2 is the usual bar.
- **Sharpe** shows what the skill is worth once turned into a long/short portfolio with costs.
- Gradient boosting (GBM) can capture non-linear effects, such as "hedge funds extremely long", that a linear model (Ridge) cannot.
""")
    gbm_sets = [c for c in ["GBM [price only]", "GBM [positioning only]", "GBM"] if c in ic.columns]
    if gbm_sets:
        st.markdown("### Is the skill stable over time?")
        roll = ic[gbm_sets].rolling(24).mean().rename(columns={"GBM [price only]": "Price only",
                                                               "GBM [positioning only]": "Positioning only",
                                                               "GBM": "Price + positioning"})
        st.plotly_chart(px.line(roll, color_discrete_map=SET_COLORS,
                                labels={"value": "Rank IC (24-month average)", "index": "", "variable": ""},
                                title="Gradient boosting: rolling 24-month IC")
                        .add_hline(y=0, line_color=INK, line_width=1)
                        .update_layout(height=360, legend=dict(orientation="h", y=-0.15)),
                        width="stretch")
        st.caption("A signal that is positive only in a few periods is fragile; a durable edge stays above zero.")

# ------------------------------------------------------------------ performance
with tab_perf:
    st.markdown("### How would each strategy have performed?")
    best = table.loc[[s for s in ML if s in table.index], "Sharpe"].idxmax()
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Best ML model (long/short)", best,
              help="Best of the machine-learning models only. Benchmarks such as the long-only basket are not "
                   "candidates: they are references to beat.")
    c2.metric("Sharpe (OOS)", f"{table.loc[best, 'Sharpe']:.2f}",
              f"{table.loc[best, 'Sharpe'] - table.loc['EW long-only', 'Sharpe']:+.2f} vs long-only")
    c3.metric("Annual return", f"{table.loc[best, 'Ann. return']:.1%}")
    c4.metric("Worst drawdown", f"{table.loc[best, 'Max DD']:.1%}")
    c5.metric("Mean IC", f"{table.loc[best, 'Mean IC']:.3f}" if pd.notna(table.loc[best].get("Mean IC")) else "–")

    st.caption("The cards above describe the best **machine-learning** model. The equal-weight long-only basket is a "
               "market reference, not a model: it bets on commodities going up, while the long/short strategies bet on "
               "some commodities beating others and are roughly market-neutral.")
    st.info("Returns are those of single-commodity ETFs (USO, GLD, CORN…): they include the cost of rolling futures "
            "contracts, so they are what an investor would actually have earned. The trade-off is a smaller "
            "universe and a history starting around 2010.", icon="ℹ️")
    default = [s for s in ["Ensemble", best, "Momentum 12-1", "Hedging pressure", "Managed money contrarian",
                           "Carry", "EW long-only"] if s in rets.columns]
    sel = st.multiselect("Strategies to compare", strategies_all, default=list(dict.fromkeys(default)))
    wealth = (1 + rets[sel].fillna(0)).cumprod()
    log_scale = st.toggle("Log scale", value=True)
    fig = px.line(wealth, labels={"value": "Value of 1 invested", "index": "", "variable": ""}, log_y=log_scale,
                  title="Growth of 1 unit invested (net of costs)")
    fig.add_hline(y=1, line_color="#CBC3B3", line_dash="dot")
    fig.update_layout(height=440, legend=dict(orientation="h", y=-0.15))
    st.plotly_chart(fig, width="stretch")

    dd = wealth / wealth.cummax() - 1
    fig_dd = px.area(dd, labels={"value": "Drawdown", "index": "", "variable": ""},
                     title="Drawdown: loss from the previous peak")
    fig_dd.update_traces(stackgroup=None, fill="tozeroy", line_width=1)
    fig_dd.update_layout(height=260, showlegend=False, yaxis_tickformat=".0%")
    st.plotly_chart(fig_dd, width="stretch")
    how_to_read("""
- Each line is a **long/short** portfolio rebalanced monthly: long the best-ranked commodities, short the worst.
  A flat line around 1 means the signal earns nothing; a rising line means it adds value.
- **EW long-only** simply buys every commodity in equal weight, as a market reference.
- **Momentum 12-1**, **Hedging pressure** and **Managed money contrarian** are classic one-signal rules from
  the academic literature: the ML models have to beat them to justify their complexity.
""")

    st.markdown("### Full statistics")
    fmt = {c: "{:.1%}" for c in ["Ann. return", "Ann. vol", "Max DD", "Hit ratio"]}
    fmt.update({c: "{:.2f}" for c in ["Sharpe", "Sortino", "Avg turnover", "t-stat", "IC t-stat"]})
    fmt["Mean IC"] = "{:.3f}"
    st.dataframe(table.style.format(fmt, na_rep="–").background_gradient(subset=["Sharpe"], cmap="RdYlGn"),
                 width="stretch")
    st.caption("Hit ratio = share of positive months. Turnover = share of the book traded each month (2 = full flip).")

    yearly = rets[sel].groupby(rets.index.year).apply(lambda x: (1 + x).prod() - 1)
    st.markdown("### Calendar-year returns")
    st.dataframe(yearly.style.format("{:.1%}").background_gradient(cmap="RdYlGn", axis=None), width="stretch")

# ------------------------------------------------------------------ robustness
with tab_rob:
    st.markdown("### Would the conclusions survive small changes?")
    st.markdown("A real signal should not depend on one lucky period, on unrealistically low costs or on one "
                "precise portfolio rule. Each table below changes **one** ingredient and recomputes the strategies.")
    cand = [c for c in ["Ridge [price only]", "Hedging pressure", "RandomForest", "Ensemble",
                        "Ridge [positioning only]", "Momentum 12-1", "EW long-only"] if c in strategies_all]
    pick = st.multiselect("Strategies to stress-test", strategies_all, default=cand[:5])
    split = pd.Timestamp("2020-01-01")

    st.markdown("**1. By sub-period** (current settings)")
    per = pd.DataFrame({
        "Sharpe before 2020": bt.summary(rets[rets.index < split], turns)["Sharpe"],
        "Sharpe 2020 onwards": bt.summary(rets[rets.index >= split], turns)["Sharpe"],
        "Sharpe full period": table["Sharpe"],
    }).loc[[p_ for p_ in pick if p_ in table.index]]
    ic_per = pd.DataFrame({
        "IC t-stat before 2020": ic[ic.index < split].pipe(
            lambda x: x.mean() / x.std() * np.sqrt(x.count())),
        "IC t-stat 2020 onwards": ic[ic.index >= split].pipe(lambda x: x.mean() / x.std() * np.sqrt(x.count())),
    })
    per = per.join(ic_per, how="left")
    st.dataframe(per.style.format("{:.2f}").background_gradient(cmap="RdYlGn", vmin=-1, vmax=1, axis=None),
                 width="stretch")

    st.markdown("**2. By transaction cost** (Sharpe ratio)")
    cost_rows = {}
    for c_ in [0, 5, 10, 20, 30]:
        r_, t_ = run_portfolios(str(RES), q, c_, inv_vol, 0.10 if vt else None)
        cost_rows[f"{c_} bps"] = bt.summary(r_, t_)["Sharpe"]
    costs = pd.DataFrame(cost_rows).loc[[p_ for p_ in pick if p_ in table.index]]
    c1, c2 = st.columns([3, 2])
    c1.plotly_chart(px.line(costs.T, markers=True, labels={"value": "Sharpe ratio", "index": "Cost", "variable": ""},
                            title="Sharpe ratio as trading gets more expensive")
                    .add_hline(y=0, line_color=INK, line_width=1)
                    .update_layout(height=360, legend=dict(orientation="h", y=-0.2)), width="stretch")
    c2.dataframe(costs.style.format("{:.2f}").background_gradient(cmap="RdYlGn", vmin=-1, vmax=1, axis=None),
                 width="stretch")

    st.markdown("**3. By portfolio rule** (Sharpe ratio for different bucket sizes)")
    rule_rows = {}
    for q_, lab in [(0.25, "Quartiles"), (1 / 3, "Terciles"), (0.5, "Halves")]:
        r_, t_ = run_portfolios(str(RES), q_, tc, inv_vol, 0.10 if vt else None)
        rule_rows[lab] = bt.summary(r_, t_)["Sharpe"]
    st.dataframe(pd.DataFrame(rule_rows).loc[[p_ for p_ in pick if p_ in table.index]].style.format("{:.2f}")
                 .background_gradient(cmap="RdYlGn", vmin=-1, vmax=1, axis=None), width="stretch")

    n_tests = int(table["IC t-stat"].notna().sum())
    st.markdown("**4. The multiple-testing trap**")
    st.markdown(f"This site compares **{n_tests} signals**. Even if none had any real skill, the chance that at least "
                f"one shows |t| > 2 purely by luck is about **{1 - 0.954 ** n_tests:.0%}**. A stricter bar that "
                f"accounts for this (Bonferroni) is roughly **t > 2.9**. A signal that is positive in both "
                "sub-periods, survives realistic costs and clears a high t-stat is far more credible than one good number.")
    how_to_read("""
- Green = Sharpe above zero, red = below. Look for strategies that stay **green across columns**.
- The sub-period split (before / after 2020) separates a calm decade from the Covid and energy-crisis years.
- Commodity ETFs are liquid: 5–10 bps per unit traded is a realistic cost; 30 bps is a pessimistic stress case.
""")

# ------------------------------------------------------------------ live book
with tab_live:
    st.markdown(f"### What the models would hold for the month after {meta['signal_date']}")
    st.markdown("The **ensemble** averages the four ML models. Commodities at the top are bought, those at the "
                "bottom are sold; the middle is not traded.")
    lv = live.copy()
    lv["side"] = np.sign(lv["weight"]).map({1: "LONG", -1: "SHORT", 0: "–"})
    lv_sorted = lv.sort_values("Ensemble")
    fig = px.bar(lv_sorted, x="Ensemble", y="name", color="sector", orientation="h",
                 color_discrete_map=SECTOR_COLORS, labels={"Ensemble": "Ensemble score (0 = worst, 1 = best)", "name": ""})
    # plotly draws the first category at the bottom: ascending scores -> highest on top, across sectors
    fig.update_yaxes(categoryorder="array", categoryarray=lv_sorted["name"].tolist())
    fig.update_layout(height=680, legend=dict(orientation="h", y=-0.08, title=""))
    c1, c2 = st.columns([3, 2])
    c1.plotly_chart(fig, width="stretch")
    show = lv[["name", "sector", "side", "weight"] + [m for m in ML if m in lv]].sort_values("weight", ascending=False)
    c2.dataframe(show.style.format({"weight": "{:+.1%}", **{m: "{:.2f}" for m in ML}})
                 .map(lambda v: "color:#2F9E44;font-weight:600" if v == "LONG"
                      else ("color:#D6336C;font-weight:600" if v == "SHORT" else ""), subset=["side"]),
                 width="stretch", height=680)
    expo = lv.groupby("sector")["weight"].sum().rename("Net weight").reset_index()
    st.plotly_chart(px.bar(expo, x="sector", y="Net weight", color="sector", color_discrete_map=SECTOR_COLORS,
                           title="Net exposure by sector (positive = net long)", labels={"sector": ""})
                    .update_layout(showlegend=False, height=320, yaxis_tickformat=".0%"), width="stretch")
    st.caption("Educational illustration of the model output, not a trading recommendation.")

# ------------------------------------------------------------------ diagnostics
with tab_diag:
    st.markdown("### Under the hood")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Prediction skill over time (rolling 12-month IC)**")
        icsel = st.multiselect("Signals", list(ic.columns),
                               default=[c for c in ["Ensemble", "Momentum 12-1", "Managed money contrarian"] if c in ic.columns])
        st.plotly_chart(px.line(ic[icsel].rolling(12).mean(),
                                labels={"value": "Rank IC", "index": "", "variable": ""})
                        .add_hline(y=0, line_color=INK, line_width=1)
                        .update_layout(height=420, legend=dict(orientation="h", y=-0.15)), width="stretch")
    with c2:
        st.markdown(f"**Which signals matter most? ({meta.get('importance_model', 'best model')})**")
        d = imp.sort_values().rename("IC drop").to_frame()
        d["Group"] = [feature_group(f) for f in d.index]
        d.index = [f"{f} — {FEATURE_DOC.get(f, '')}" for f in d.index]
        st.plotly_chart(px.bar(d, x="IC drop", y=d.index, color="Group", orientation="h",
                               color_discrete_map={"Price": "#1C7ED6", "Positioning": "#E8590C"}, labels={"y": ""})
                        .update_layout(height=560, legend=dict(orientation="h", y=-0.08, title="")), width="stretch")
    how_to_read("""
- **Feature importance**: each signal is shuffled at random, and we measure how much the model's IC drops.
  A large drop means the model relies on that signal. Blue = price signals, orange = positioning signals.
- **Rolling IC**: the skill of each signal year by year. Skill that comes and goes is typical of financial data.
""")
    st.markdown("**Do the signals say the same thing? (average rank correlation)**")
    sigs = [c for c in P.columns if c != "fwd_ret"]
    corr = P[sigs].groupby(level="date").corr(method="spearman").groupby(level=1).mean().loc[sigs, sigs]
    st.plotly_chart(px.imshow(corr, text_auto=".2f", color_continuous_scale="RdBu", zmin=-1, zmax=1)
                    .update_layout(height=560), width="stretch")
    st.caption("Close to 1 = two signals rank commodities the same way; close to 0 = they carry different information.")
    st.markdown("**Trading intensity (monthly turnover, 12-month average)**")
    st.plotly_chart(px.line(turns.rolling(12).mean(), labels={"value": "Turnover", "index": "", "variable": ""})
                    .update_layout(height=320, legend=dict(orientation="h", y=-0.2)), width="stretch")

# ------------------------------------------------------------------ factor explorer
with tab_fact:
    st.markdown("### Explore one signal at a time")
    st.markdown("Pick a signal: every month the commodities are sorted into buckets by that signal, and we look at "
                "their average return the **following** month. A staircase from left to right means the signal works.")
    feats = meta["features"]
    c1, c2 = st.columns([1, 3])
    f = c1.selectbox("Signal", feats, format_func=lambda x: f"{x} — {FEATURE_DOC.get(x, '')}")
    c1.markdown(f"<span class='pill' style='background:{'#FFE8CC' if feature_group(f) == 'Positioning' else '#D0EBFF'}'>"
                f"{feature_group(f)} signal</span>", unsafe_allow_html=True)
    nq = c1.radio("Sort into", [3, 5], format_func=lambda n: f"{n} buckets", horizontal=True)
    d = panel.dropna(subset=["fwd_ret", f"raw_{f}"])[[f, "fwd_ret"]].copy()   # ignore neutral-filled gaps
    d["bucket"] = np.ceil(d[f] * nq).clip(1, nq).astype(int)
    port = d.groupby([d.index.get_level_values("date"), "bucket"])["fwd_ret"].mean().unstack()
    ann = port.mean() * 12
    hml = port.iloc[:, -1] - port.iloc[:, 0]
    c1.metric("High minus low (per year)", f"{ann.iloc[-1] - ann.iloc[0]:.1%}")
    c1.metric("t-stat", f"{hml.mean() / hml.std() * np.sqrt(hml.count()):.2f}", help="Above 2 in absolute value = significant.")
    colors = [PALETTE[1] if v >= 0 else PALETTE[6] for v in ann.values]
    fig = go.Figure(go.Bar(x=[f"Q{i}" + (" (low)" if i == 1 else " (high)" if i == nq else "") for i in ann.index],
                           y=ann.values, marker_color=colors))
    fig.update_layout(title="Average next-month return by bucket (annualised)", yaxis_tickformat=".1%", height=360)
    c2.plotly_chart(fig, width="stretch")

    st.markdown("### One commodity's history")
    tickers_in_data = list(dict.fromkeys(panel.index.get_level_values("ticker")))
    tk = st.selectbox("Commodity", tickers_in_data,
                      format_func=lambda t: f"{NAMES.get(t, t)} ({SECTOR.get(t, '')})")
    raw = panel.xs(tk, level="ticker")
    rawcols = [c for c in raw.columns if c.startswith("raw_")]
    pick = st.multiselect("Signals to plot", rawcols, format_func=lambda c: c.replace("raw_", ""),
                          default=[c for c in ["raw_mm_net", "raw_pm_net", "raw_hp"] if c in rawcols])
    st.plotly_chart(px.line(raw[pick].rename(columns=lambda c: c.replace("raw_", "")),
                            labels={"value": "", "date": "", "variable": ""})
                    .update_layout(height=360, legend=dict(orientation="h", y=-0.15)), width="stretch")
    st.caption("mm_net = hedge funds' net position, pm_net = producers' net position, hp = hedgers' long share "
               "(all as a share of open interest).")

# ------------------------------------------------------------------ methodology
with tab_method:
    st.markdown((ROOT / "docs" / "methodology.md").read_text())

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
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from cfl import backtest as bt                  # noqa: E402
from cfl.config import SECTOR, UNIVERSE         # noqa: E402
from cfl.features import FEATURE_DOC            # noqa: E402

st.set_page_config(page_title="Commodity Factor Lab", page_icon="🛢️", layout="wide")

LIVE, DEMO = ROOT / "data" / "results", ROOT / "data" / "results_demo"
RES = LIVE if (LIVE / "meta.json").exists() else DEMO
NAMES = {c.yf: c.name for c in UNIVERSE}
ML = ["Ridge", "RandomForest", "GBM", "MLP", "Ensemble"]
PALETTE = px.colors.qualitative.Safe


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
    st.title("🛢️ Commodity Factor Lab")
    st.caption("Machine learning for cross-sectional commodity futures returns")
    if RES == DEMO:
        st.warning("Running on **synthetic demo data**. Run the pipeline with `--source yahoo` for live results.")
    st.markdown(f"**Signal date:** {meta['signal_date']}  \n**Last run:** {meta['run_utc'][:16]} UTC  \n"
                f"**Universe:** {meta['n_assets']} futures  \n**Data:** {meta['source']}"
                f"{' + CFTC COT' if meta['cot'] else ''}{' + carry' if meta['carry'] else ''}")
    st.divider()
    st.subheader("Portfolio construction")
    q = st.select_slider("Long/short bucket", options=[0.2, 0.25, 1 / 3, 0.5], value=1 / 3,
                         format_func=lambda x: {0.2: "Quintiles", 0.25: "Quartiles", 1/3: "Terciles", 0.5: "Halves"}[x])
    tc = st.slider("Transaction cost (bps per unit turnover)", 0, 30, 5)
    inv_vol = st.toggle("Inverse-volatility weights", value=True)
    vt = st.toggle("Target 10% volatility", value=True)
    st.divider()
    st.caption("Built by Hugo Guidon — emlyon MSc in Finance. Educational project, not investment advice.")

rets, turns = run_portfolios(str(RES), q, tc, inv_vol, 0.10 if vt else None)
ic_cols = [c for c in ic.columns]
table = bt.summary(rets, turns, ic)

tab_perf, tab_live, tab_diag, tab_fact, tab_method = st.tabs(
    ["📈 Performance", "🎯 This month's book", "🔬 Model diagnostics", "🧭 Factor explorer", "📚 Methodology"])

# ------------------------------------------------------------------ performance
with tab_perf:
    best = table.loc[[s for s in ML if s in table.index], "Sharpe"].idxmax()
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Best ML model", best)
    c2.metric("Sharpe (OOS)", f"{table.loc[best, 'Sharpe']:.2f}",
              f"{table.loc[best, 'Sharpe'] - table.loc['EW long-only', 'Sharpe']:+.2f} vs EW long")
    c3.metric("Ann. return", f"{table.loc[best, 'Ann. return']:.1%}")
    c4.metric("Max drawdown", f"{table.loc[best, 'Max DD']:.1%}")
    c5.metric("Mean IC", f"{table.loc[best, 'Mean IC']:.3f}" if pd.notna(table.loc[best].get("Mean IC")) else "–")

    default = [s for s in ["Ensemble", best, "Momentum 12-1", "Hedging pressure", "Carry", "EW long-only"]
               if s in rets.columns]
    sel = st.multiselect("Strategies", strategies_all, default=list(dict.fromkeys(default)))
    wealth = (1 + rets[sel].fillna(0)).cumprod()
    log_scale = st.toggle("Log scale", value=True)
    fig = px.line(wealth, labels={"value": "Growth of 1", "index": "", "variable": ""},
                  color_discrete_sequence=PALETTE, log_y=log_scale)
    fig.update_layout(height=430, legend=dict(orientation="h", y=1.08), margin=dict(t=30, b=10))
    st.plotly_chart(fig, use_container_width=True)

    dd = wealth / wealth.cummax() - 1
    fig_dd = px.area(dd, labels={"value": "Drawdown", "index": "", "variable": ""},
                     color_discrete_sequence=PALETTE)
    fig_dd.update_traces(stackgroup=None, fill="tozeroy")
    fig_dd.update_layout(height=250, showlegend=False, margin=dict(t=10, b=10), yaxis_tickformat=".0%")
    st.plotly_chart(fig_dd, use_container_width=True)

    st.subheader("Out-of-sample statistics")
    fmt = {c: "{:.1%}" for c in ["Ann. return", "Ann. vol", "Max DD", "Hit ratio"]}
    fmt.update({c: "{:.2f}" for c in ["Sharpe", "Sortino", "Avg turnover", "t-stat", "IC t-stat"]})
    fmt["Mean IC"] = "{:.3f}"
    st.dataframe(table.style.format(fmt, na_rep="–").background_gradient(subset=["Sharpe"], cmap="RdYlGn"),
                 use_container_width=True)

    yearly = rets[sel].groupby(rets.index.year).apply(lambda x: (1 + x).prod() - 1)
    st.subheader("Calendar-year returns")
    st.dataframe(yearly.style.format("{:.1%}").background_gradient(cmap="RdYlGn", axis=None),
                 use_container_width=True)

# ------------------------------------------------------------------ live book
with tab_live:
    st.subheader(f"Ensemble positions for the month after {meta['signal_date']}")
    st.caption("Long the top bucket, short the bottom bucket of the ensemble score. Weights per leg sum to 100% (before vol targeting).")
    lv = live.copy()
    lv["side"] = np.sign(lv["weight"]).map({1: "LONG", -1: "SHORT", 0: "–"})
    fig = px.bar(lv.sort_values("Ensemble"), x="Ensemble", y="name", color="sector", orientation="h",
                 color_discrete_sequence=PALETTE, labels={"Ensemble": "Ensemble score (rank)", "name": ""})
    fig.update_layout(height=650, margin=dict(t=10))
    c1, c2 = st.columns([3, 2])
    c1.plotly_chart(fig, use_container_width=True)
    show = lv[["name", "sector", "side", "weight"] + [m for m in ML if m in lv]].sort_values("weight", ascending=False)
    c2.dataframe(show.style.format({"weight": "{:+.1%}", **{m: "{:.2f}" for m in ML}}),
                 use_container_width=True, height=650)
    expo = lv.groupby("sector")["weight"].sum()
    st.plotly_chart(px.bar(expo, labels={"value": "Net weight", "sector": ""}, title="Net sector exposure",
                           color_discrete_sequence=PALETTE).update_layout(showlegend=False, height=300,
                                                                          yaxis_tickformat=".0%"),
                    use_container_width=True)

# ------------------------------------------------------------------ diagnostics
with tab_diag:
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Rolling 12-month IC")
        icsel = st.multiselect("Signals", ic_cols, default=[c for c in ["Ensemble", "Momentum 12-1", "Hedging pressure"] if c in ic_cols])
        st.plotly_chart(px.line(ic[icsel].rolling(12).mean(), color_discrete_sequence=PALETTE,
                                labels={"value": "Spearman IC", "index": "", "variable": ""})
                        .add_hline(y=0, line_dash="dot").update_layout(height=380, legend=dict(orientation="h")),
                        use_container_width=True)
    with c2:
        st.subheader(f"Feature importance ({meta.get('importance_model', 'best model')})")
        st.caption("Drop in mean OOS IC when the feature is shuffled within each month.")
        d = imp.rename(index=lambda f: f"{f} — {FEATURE_DOC.get(f, '')}").sort_values()
        st.plotly_chart(px.bar(d, orientation="h", color_discrete_sequence=PALETTE[1:],
                               labels={"value": "IC drop", "index": ""})
                        .update_layout(showlegend=False, height=380, margin=dict(l=10)),
                        use_container_width=True)
    st.subheader("Signal correlation (average cross-sectional rank correlation)")
    sigs = [c for c in P.columns if c != "fwd_ret"]
    corr = P[sigs].groupby(level="date").corr(method="spearman").groupby(level=1).mean().loc[sigs, sigs]
    st.plotly_chart(px.imshow(corr, text_auto=".2f", color_continuous_scale="RdBu", zmin=-1, zmax=1)
                    .update_layout(height=480), use_container_width=True)
    st.subheader("Monthly turnover")
    st.plotly_chart(px.line(turns.rolling(12).mean(), color_discrete_sequence=PALETTE,
                            labels={"value": "Turnover (12m avg)", "index": "", "variable": ""})
                    .update_layout(height=300), use_container_width=True)

# ------------------------------------------------------------------ factor explorer
with tab_fact:
    feats = meta["features"]
    c1, c2 = st.columns([1, 3])
    f = c1.selectbox("Characteristic", feats, format_func=lambda x: f"{x} — {FEATURE_DOC.get(x, '')}")
    nq = c1.radio("Sort into", [3, 5], format_func=lambda n: f"{n} portfolios", horizontal=True)
    d = panel.dropna(subset=["fwd_ret", f"raw_{f}"])[[f, "fwd_ret"]].copy()   # ignore neutral-filled gaps
    d["bucket"] = np.ceil(d[f] * nq).clip(1, nq).astype(int)
    port = d.groupby([d.index.get_level_values("date"), "bucket"])["fwd_ret"].mean().unstack()
    ann = port.mean() * 12
    c1.metric("High-minus-low (ann.)", f"{ann.iloc[-1] - ann.iloc[0]:.1%}")
    hml = port.iloc[:, -1] - port.iloc[:, 0]
    c1.metric("HML t-stat", f"{hml.mean() / hml.std() * np.sqrt(hml.count()):.2f}")
    fig = go.Figure(go.Bar(x=[f"Q{i}" for i in ann.index], y=ann.values, marker_color=PALETTE[0]))
    fig.update_layout(title="Average annualised next-month return by bucket", yaxis_tickformat=".1%", height=330)
    c2.plotly_chart(fig, use_container_width=True)
    tk = st.selectbox("Commodity", list(NAMES), format_func=lambda t: f"{NAMES[t]} ({SECTOR[t]})")
    raw = panel.xs(tk, level="ticker")
    rawcols = [c for c in raw.columns if c.startswith("raw_")]
    pick = st.multiselect("Raw characteristics", rawcols, default=[c for c in ["raw_hp", "raw_spec_net", "raw_mom_12_1"] if c in rawcols])
    st.plotly_chart(px.line(raw[pick], color_discrete_sequence=PALETTE,
                            labels={"value": "", "date": "", "variable": ""}).update_layout(height=350),
                    use_container_width=True)

# ------------------------------------------------------------------ methodology
with tab_method:
    st.markdown((ROOT / "docs" / "methodology.md").read_text())

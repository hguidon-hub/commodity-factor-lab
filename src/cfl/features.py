"""Monthly cross-sectional feature panel.

Every feature is known at the month-end close t; the label is the return over month t+1.
Features are uniformised cross-sectionally (rank -> [0, 1]) as in Coqueret & Guida (2020),
which removes scale and outlier issues and makes all models comparable.

Two feature groups drive the research question ("does positioning add information?"):
  PRICE        : built from prices only
  POSITIONING  : built from the CFTC Commitments of Traders (legacy + disaggregated reports)
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import SECTOR
from .data import clean_roll_jumps

PRICE_FEATURES = ["rev_1m", "mom_3m", "mom_6m", "mom_12_1", "vol_3m", "skew_6m", "max_1m",
                  "carry", "carry_chg_3m"]          # carry only with a 2nd-contract source
LEGACY_COT = ["hp", "hp_z3y", "spec_net", "spec_net_chg_3m", "oi_chg_3m"]
DISAGG_COT = ["mm_net", "mm_net_z3y", "mm_net_chg_1m", "mm_crowding", "pm_net", "pm_net_z3y",
              "swap_net", "conc4_long", "conc4_net"]
POSITIONING_FEATURES = LEGACY_COT + DISAGG_COT
ALL_FEATURES = PRICE_FEATURES + POSITIONING_FEATURES

FEATURE_DOC = {
    "rev_1m": "Last month return (short-term reversal)",
    "mom_3m": "3-month momentum",
    "mom_6m": "6-month momentum",
    "mom_12_1": "12-month momentum skipping the last month",
    "vol_3m": "Realised volatility, 3 months (annualised)",
    "skew_6m": "Skewness of daily returns, 6 months",
    "max_1m": "Max daily return last month (lottery effect)",
    "carry": "Roll yield ln(F1/F2) (needs 2nd contract)",
    "carry_chg_3m": "3-month change in carry",
    "hp": "Hedging pressure: commercials' long share",
    "hp_z3y": "Hedging pressure, z-score vs own 3y history",
    "spec_net": "Non-commercials' net position / open interest",
    "spec_net_chg_3m": "3-month change in non-commercials' net position",
    "oi_chg_3m": "3-month log change in open interest",
    "mm_net": "Managed money (hedge funds, CTAs) net position / OI",
    "mm_net_z3y": "Managed money net, z-score vs own 3y history",
    "mm_net_chg_1m": "1-month change in managed money net position",
    "mm_crowding": "Managed money net, percentile of own 3y range (crowding)",
    "pm_net": "Producers & merchants net position / OI",
    "pm_net_z3y": "Producers & merchants net, z-score vs own 3y history",
    "swap_net": "Swap dealers (index & OTC flows) net position / OI",
    "conc4_long": "Share of longs held by the 4 largest traders",
    "conc4_net": "Top-4 long share minus top-4 short share",
}


def feature_group(f: str) -> str:
    return "Positioning" if f in POSITIONING_FEATURES else "Price"


def _month_ends(idx: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """Last trading day of each month; the current month is dropped while still incomplete."""
    s = pd.Series(idx, index=idx)
    me = pd.DatetimeIndex(s.groupby(idx.to_period("M")).max().values)
    if len(me) and (pd.offsets.BMonthEnd().rollforward(me[-1]) - me[-1]).days > 3:
        me = me[:-1]
    return me


def price_features(px: pd.DataFrame, px2: pd.DataFrame | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (features long-format, monthly returns wide)."""
    r = clean_roll_jumps(px)
    me = _month_ends(r.dropna(how="all").index)
    logr = np.log1p(r.fillna(0.0))
    cum = logr.cumsum()
    cum_m = cum.reindex(me)
    valid = px.notna().reindex(me)

    ret_m = np.expm1(cum_m.diff()).where(valid)            # simple return over month t
    feats = {
        "rev_1m": ret_m,
        "mom_3m": np.expm1(cum_m.diff(3)),
        "mom_6m": np.expm1(cum_m.diff(6)),
        "mom_12_1": np.expm1(cum_m.shift(1).diff(11)),
        "vol_3m": r.rolling(63, min_periods=40).std().reindex(me) * np.sqrt(252),
        "skew_6m": r.rolling(126, min_periods=80).skew().reindex(me),
        "max_1m": r.rolling(21, min_periods=15).max().reindex(me),
    }
    if px2 is not None:
        carry = np.log(px / px2.reindex_like(px)).reindex(me)
        feats["carry"] = carry
        feats["carry_chg_3m"] = carry.diff(3)
    panel = pd.concat({k: v.where(valid) for k, v in feats.items()}, axis=1)
    long = panel.stack(level=1, future_stack=True).rename_axis(["date", "ticker"])
    return long, ret_m


def _z(s: pd.Series, n: int = 36) -> pd.Series:
    m = s.rolling(n, min_periods=n // 2)
    return (s - m.mean()) / m.std()


def _pct_range(s: pd.Series, n: int = 36) -> pd.Series:
    lo, hi = s.rolling(n, min_periods=n // 2).min(), s.rolling(n, min_periods=n // 2).max()
    return (s - lo) / (hi - lo)


def cot_features(cot: pd.DataFrame, month_ends: pd.DatetimeIndex) -> pd.DataFrame:
    """COT positions as known at each month end (report of Tuesday, public on Friday => +3 days)."""
    c = cot.copy()
    c["avail"] = pd.to_datetime(c["date"]) + pd.Timedelta(days=3)
    raw_cols = []
    if {"comm_long", "comm_short", "spec_long", "spec_short", "oi"} <= set(c.columns):
        c["hp"] = c["comm_long"] / (c["comm_long"] + c["comm_short"])
        c["spec_net"] = (c["spec_long"] - c["spec_short"]) / c["oi"]
        c["log_oi"] = np.log(c["oi"])
        raw_cols += ["hp", "spec_net", "log_oi"]
    disagg = {"mm_long", "mm_short", "pm_long", "pm_short", "swap_long", "swap_short", "oi_d"} <= set(c.columns)
    if disagg:
        c["mm_net"] = (c["mm_long"] - c["mm_short"]) / c["oi_d"]
        c["pm_net"] = (c["pm_long"] - c["pm_short"]) / c["oi_d"]
        c["swap_net"] = (c["swap_long"] - c["swap_short"]) / c["oi_d"]
        c["conc4_long"] = c["conc4_long"] / 100
        c["conc4_net"] = c["conc4_long"] - c["conc4_short"] / 100
        raw_cols += ["mm_net", "pm_net", "swap_net", "conc4_long", "conc4_net"]

    out = []
    for tk, g in c.groupby("ticker"):
        g = g.set_index("avail").sort_index()[raw_cols]
        g = g[~g.index.duplicated(keep="last")]
        m = g.reindex(g.index.union(month_ends)).ffill(limit=10).reindex(month_ends)
        if "hp" in m:
            m["hp_z3y"] = _z(m["hp"])
            m["spec_net_chg_3m"] = m["spec_net"].diff(3)
            m["oi_chg_3m"] = m["log_oi"].diff(3)
            m = m.drop(columns="log_oi")
        if disagg:
            m["mm_net_z3y"] = _z(m["mm_net"])
            m["mm_net_chg_1m"] = m["mm_net"].diff(1)
            m["mm_crowding"] = _pct_range(m["mm_net"])
            m["pm_net_z3y"] = _z(m["pm_net"])
        m["ticker"] = tk
        out.append(m)
    f = pd.concat(out).rename_axis("date").reset_index().set_index(["date", "ticker"])
    return f.replace([np.inf, -np.inf], np.nan)


def uniformise(df: pd.DataFrame) -> pd.DataFrame:
    """Cross-sectional rank to (0, 1] per date; missing -> 0.5 (neutral)."""
    u = df.groupby(level="date").rank(pct=True)
    return u.fillna(0.5)


def build_panel(px: pd.DataFrame, px2: pd.DataFrame | None = None,
                cot: pd.DataFrame | None = None) -> pd.DataFrame:
    """Full modelling panel indexed by (date, ticker).

    Columns: raw features (prefixed 'raw_'), uniformised features, sector,
             fwd_ret (label, month t+1 return), fwd_rank (uniformised label).
    attrs: features (all), price_features, positioning_features.
    """
    pf, ret_m = price_features(px, px2)
    me = ret_m.index
    panel = pf
    if cot is not None and len(cot):
        panel = panel.join(cot_features(cot, me), how="left")
    feats = [f for f in ALL_FEATURES if f in panel.columns and panel[f].notna().any()]
    panel = panel[feats]

    fwd = ret_m.shift(-1).stack(future_stack=True).rename("fwd_ret").rename_axis(["date", "ticker"])
    panel = panel.join(fwd, how="left")
    # keep rows where the asset trades (has a price this month)
    panel = panel[panel["rev_1m"].notna()]
    panel = panel[panel.groupby(level="date")["rev_1m"].transform("count") >= 8]   # min breadth

    out = uniformise(panel[feats])
    out = out.join(panel[feats].add_prefix("raw_"))
    out["sector"] = out.index.get_level_values("ticker").map(SECTOR)
    out["fwd_ret"] = panel["fwd_ret"]
    out["fwd_rank"] = panel["fwd_ret"].groupby(level="date").rank(pct=True)
    out.attrs["features"] = feats
    out.attrs["price_features"] = [f for f in feats if feature_group(f) == "Price"]
    out.attrs["positioning_features"] = [f for f in feats if feature_group(f) == "Positioning"]
    return out.sort_index()

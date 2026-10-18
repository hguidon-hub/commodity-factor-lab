"""Walk-forward (expanding window) training and long/short portfolio backtest."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.base import clone

from .config import MIN_TRAIN_MONTHS, TC_BPS
from .models import BENCHMARKS, model_zoo


# --------------------------------------------------------------------------- predictions
def walk_forward(panel: pd.DataFrame, models: dict | None = None, refit_every: int = 12,
                 min_train: int = MIN_TRAIN_MONTHS, label: str = "fwd_rank",
                 window: int | None = None, verbose: bool = False) -> pd.DataFrame:
    """Out-of-sample predictions for every model.

    At month t the model is trained on all (features, label) pairs with date < t, i.e. labels
    that are fully realised at t. Retrained every `refit_every` months (expanding window, or
    rolling if `window` is set). The last month (label unknown) gets a live prediction.
    """
    feats = panel.attrs["features"]
    models = models or model_zoo()
    dates = panel.index.get_level_values("date").unique().sort_values()
    preds = []
    fitted: dict = {}
    for i, t in enumerate(dates):
        if i < min_train:
            continue
        if (i - min_train) % refit_every == 0:
            lo = dates[max(0, i - window)] if window else dates[0]
            tr = panel.loc[(panel.index.get_level_values("date") >= lo)
                           & (panel.index.get_level_values("date") < t)].dropna(subset=[label])
            X, y = tr[feats].values, tr[label].values
            fitted = {k: clone(m).fit(X, y) for k, m in models.items()}
            last_refit = t
            if verbose:
                print(f"refit {t:%Y-%m}  n_train={len(tr)}")
        te = panel.xs(t, level="date", drop_level=False)
        out = pd.DataFrame(index=te.index)
        for k, m in fitted.items():
            out[k] = m.predict(te[feats].values)
        preds.append(out)
    P = pd.concat(preds)
    # ensemble = average of cross-sectional ranks (robust to different output scales)
    ranks = P.groupby(level="date").rank(pct=True)
    P["Ensemble"] = ranks[list(models)].mean(axis=1)
    for name, (col, sign) in BENCHMARKS.items():
        if col in panel.columns:
            P[name] = sign * panel.loc[P.index, col]
    P["fwd_ret"] = panel.loc[P.index, "fwd_ret"]
    P.attrs["last_models"] = fitted
    P.attrs["last_refit"] = last_refit
    return P


# --------------------------------------------------------------------------- portfolios
def weights_from_signal(sig: pd.Series, q: float = 1 / 3, vol: pd.Series | None = None) -> pd.Series:
    """Long top-q / short bottom-q per date, equal or inverse-vol weighted, 100% gross each leg."""
    def one(s: pd.Series) -> pd.Series:
        s = s.dropna()
        n = max(1, int(round(len(s) * q)))
        order = s.rank(method="first")
        w = pd.Series(0.0, index=s.index)
        w[order > len(s) - n] = 1.0
        w[order <= n] = -1.0
        if vol is not None:
            iv = 1 / vol.reindex(s.index).clip(lower=0.05).fillna(vol.median())
            w = w * iv
        long, short = w[w > 0].sum(), -w[w < 0].sum()
        w[w > 0] /= long if long else 1
        w[w < 0] /= short if short else 1
        return w
    return sig.groupby(level="date", group_keys=False).apply(one)


def portfolio_returns(P: pd.DataFrame, strategies: list[str], q: float = 1 / 3,
                      tc_bps: float = TC_BPS, inv_vol: pd.Series | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Monthly net returns (dated at the END of the holding month) and turnover per strategy."""
    rets, turns = {}, {}
    realised = P.dropna(subset=["fwd_ret"])
    for s in strategies + ["EW long-only"]:
        if s == "EW long-only":
            w = realised["fwd_ret"].notna().astype(float)
            w = w / w.groupby(level="date").transform("sum")
        else:
            w = weights_from_signal(realised[s], q=q, vol=inv_vol)
        gross = (w * realised["fwd_ret"]).groupby(level="date").sum()
        W = w.unstack().fillna(0.0)
        to = W.diff().abs().sum(axis=1)
        to.iloc[0] = W.iloc[0].abs().sum()
        net = gross - to * tc_bps / 1e4
        # shift date to the end of the holding month for plotting
        net.index = net.index + pd.offsets.MonthEnd(1)
        rets[s], turns[s] = net, to
    return pd.DataFrame(rets), pd.DataFrame(turns)


def vol_target(r: pd.DataFrame, target: float = 0.10, lookback: int = 36) -> pd.DataFrame:
    """Ex-ante volatility targeting with trailing realised vol (no look-ahead), leverage capped at 3."""
    vol = r.rolling(lookback, min_periods=12).std().shift(1) * np.sqrt(12)
    lev = (target / vol).clip(upper=3.0)
    return (r * lev).dropna(how="all")


# --------------------------------------------------------------------------- metrics
def max_drawdown(r: pd.Series) -> float:
    w = (1 + r.fillna(0)).cumprod()
    return float((w / w.cummax() - 1).min())


def information_coefficient(P: pd.DataFrame, strategies: list[str]) -> pd.DataFrame:
    """Monthly Spearman rank correlation between signal and next-month return."""
    R = P.dropna(subset=["fwd_ret"])
    out = {}
    for s in strategies:
        out[s] = R.groupby(level="date").apply(
            lambda g: g[s].rank().corr(g["fwd_ret"].rank()) if g[s].notna().sum() > 3 else np.nan)
    return pd.DataFrame(out)


def summary(rets: pd.DataFrame, turns: pd.DataFrame, ic: pd.DataFrame | None = None) -> pd.DataFrame:
    rows = {}
    for s in rets:
        r = rets[s].dropna()
        ann = (1 + r).prod() ** (12 / len(r)) - 1
        vol = r.std() * np.sqrt(12)
        row = {
            "Ann. return": ann,
            "Ann. vol": vol,
            "Sharpe": r.mean() / r.std() * np.sqrt(12) if r.std() > 0 else np.nan,
            "Sortino": r.mean() / r[r < 0].std() * np.sqrt(12) if (r < 0).any() else np.nan,
            "Max DD": max_drawdown(r),
            "Hit ratio": (r > 0).mean(),
            "Avg turnover": turns[s].mean() if s in turns else np.nan,
            "t-stat": r.mean() / r.std() * np.sqrt(len(r)),
        }
        if ic is not None and s in ic:
            row["Mean IC"] = ic[s].mean()
            row["IC t-stat"] = ic[s].mean() / ic[s].std() * np.sqrt(ic[s].count())
        rows[s] = row
    return pd.DataFrame(rows).T


# --------------------------------------------------------------------------- explainability
def permutation_ic_importance(model, panel: pd.DataFrame, feats: list[str], n_repeats: int = 5,
                              seed: int = 0) -> pd.Series:
    """Drop in mean IC when a feature is shuffled within each month (OOS-style importance)."""
    rng = np.random.default_rng(seed)
    data = panel.dropna(subset=["fwd_ret"])
    def mean_ic(X: pd.DataFrame) -> float:
        p = pd.Series(model.predict(X[feats].values), index=X.index)
        return pd.concat([p.rename("p"), data["fwd_ret"]], axis=1).groupby(level="date").apply(
            lambda g: g["p"].rank().corr(g["fwd_ret"].rank())).mean()
    base = mean_ic(data)
    imp = {}
    for f in feats:
        drops = []
        for _ in range(n_repeats):
            X = data.copy()
            X[f] = X.groupby(level="date")[f].transform(lambda s: rng.permutation(s.values))
            drops.append(base - mean_ic(X))
        imp[f] = float(np.mean(drops))
    return pd.Series(imp).sort_values(ascending=False)


@dataclass
class Results:
    preds: pd.DataFrame
    rets: pd.DataFrame
    turns: pd.DataFrame
    ic: pd.DataFrame
    table: pd.DataFrame
    importance: pd.Series = field(default_factory=pd.Series)

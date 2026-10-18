"""Fast end-to-end tests on synthetic data (no network)."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from cfl import backtest as bt  # noqa: E402
from cfl.data import make_synthetic  # noqa: E402
from cfl.features import build_panel  # noqa: E402


def _panel():
    px, px2, cot = make_synthetic(n_years=10, seed=1)
    return build_panel(px, px2, cot)


def test_panel_shape_and_ranges():
    p = _panel()
    f = p.attrs["features"]
    assert {"hp", "carry", "mom_12_1"} <= set(f)
    assert p[f].min().min() > 0 and p[f].max().max() <= 1
    assert not p[f].isna().any().any()


def test_no_lookahead_label_alignment():
    """fwd_ret at t must equal the realised rev_1m at t+1 for the same asset."""
    p = _panel()
    a = p["fwd_ret"].unstack()
    b = p["raw_rev_1m"].unstack().shift(-1)
    m = a.notna() & b.notna()
    assert m.values.sum() > 100 and np.allclose(a.values[m.values], b.values[m.values])


def test_walk_forward_and_portfolio():
    p = _panel()
    P = bt.walk_forward(p, {"Ridge": Ridge(alpha=10)}, refit_every=12, min_train=36)
    assert P.index.get_level_values("date").min() > p.index.get_level_values("date").min()
    rets, turns = bt.portfolio_returns(P, ["Ridge", "Ensemble"])
    assert rets.notna().sum().min() > 40
    w = bt.weights_from_signal(P["Ridge"].dropna())
    s = w.groupby(level="date").agg(lambda x: (x[x > 0].sum(), x[x < 0].sum()))
    assert all(abs(l - 1) < 1e-9 and abs(sh + 1) < 1e-9 for l, sh in s)
    tab = bt.summary(rets, turns)
    assert "Sharpe" in tab.columns

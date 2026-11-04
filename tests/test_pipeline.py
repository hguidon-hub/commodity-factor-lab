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
    assert {"hp", "carry", "mom_12_1", "mm_net", "mm_crowding", "pm_net", "conc4_long"} <= set(f)
    assert set(p.attrs["price_features"]) | set(p.attrs["positioning_features"]) == set(f)
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


def test_ablation_uses_its_own_feature_set():
    from cfl.models import ablation_zoo
    p = _panel()
    zoo = {"Ridge": Ridge(alpha=10)} | ablation_zoo(p.attrs["price_features"], p.attrs["positioning_features"])
    P = bt.walk_forward(p, zoo, refit_every=24, min_train=36)
    fitted = P.attrs["last_models"]
    assert fitted["Ridge [price only]"][1] == p.attrs["price_features"]
    assert fitted["GBM [positioning only]"][1] == p.attrs["positioning_features"]
    assert "Ensemble" in P and P["Ridge [price only]"].notna().all()


def test_incomplete_month_dropped():
    from cfl.features import _month_ends
    idx = pd.bdate_range("2024-01-01", "2024-03-05")
    me = _month_ends(idx)
    assert me[-1] == pd.Timestamp("2024-02-29")


def test_feature_selection_in_window():
    from cfl.models import selection_zoo
    p = _panel()
    P = bt.walk_forward(p, selection_zoo(4), refit_every=24, min_train=36)
    sel = P.attrs["selections"]["Ridge [selected 4]"]
    assert sel and all(len(fs) == 4 and set(fs) <= set(p.attrs["features"]) for fs in sel.values())
    # the first selection only uses data before the first refit date
    first = min(sel)
    tr = p[p.index.get_level_values("date") < first].dropna(subset=["fwd_rank"])
    assert bt.select_features(tr, p.attrs["features"], 4) == sel[first]

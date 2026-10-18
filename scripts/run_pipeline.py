"""End-to-end pipeline: data -> features -> walk-forward ML -> backtest -> results for the app.

Usage
-----
python scripts/run_pipeline.py --source yahoo                 # live data (Yahoo + CFTC)
python scripts/run_pipeline.py --source bloomberg --bbg data/raw/bbg_px_settle.csv
python scripts/run_pipeline.py --source synthetic             # offline demo / CI test
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cfl import backtest as bt                      # noqa: E402
from cfl import data as dl                          # noqa: E402
from cfl.config import RESULTS_DIR, UNIVERSE        # noqa: E402
from cfl.features import build_panel                # noqa: E402
from cfl.models import BENCHMARKS, model_zoo        # noqa: E402


def get_data(source: str, bbg: str | None):
    px2 = None
    if source == "synthetic":
        px, px2, cot = dl.make_synthetic()
        return px, px2, cot
    if source == "bloomberg":
        px, px2 = dl.load_bloomberg_csv(bbg)
    else:
        px = dl.download_prices()
    try:
        cot = dl.download_cot()
    except Exception as e:  # CFTC down -> price-only model, flagged in meta
        print(f"[warn] COT download failed: {e}")
        cot = None
    dl.save(px, "prices")
    if cot is not None:
        cot.to_csv(dl.RAW_DIR / "cot.csv.gz", index=False)
    return px, px2, cot


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=["yahoo", "bloomberg", "synthetic"], default="yahoo")
    ap.add_argument("--bbg", help="Bloomberg CSV export (generic 1st/2nd contracts)")
    ap.add_argument("--refit-every", type=int, default=12)
    ap.add_argument("--quantile", type=float, default=1 / 3)
    ap.add_argument("--out", default=str(RESULTS_DIR))
    a = ap.parse_args(argv)

    px, px2, cot = get_data(a.source, a.bbg)
    panel = build_panel(px, px2, cot)
    feats = panel.attrs["features"]
    print(f"panel: {panel.shape}, {panel.index.get_level_values('date').nunique()} months, features={feats}")

    P = bt.walk_forward(panel, model_zoo(), refit_every=a.refit_every, verbose=True)
    strategies = list(model_zoo()) + ["Ensemble"] + [b for b in BENCHMARKS if b in P.columns]
    vol = panel["raw_vol_3m"]
    rets, turns = bt.portfolio_returns(P, strategies, q=a.quantile, inv_vol=vol)
    rets_vt = bt.vol_target(rets)
    ic = bt.information_coefficient(P, strategies)
    table = bt.summary(rets, turns, ic)
    table_vt = bt.summary(rets_vt, turns, ic)

    oos = panel[panel.index.get_level_values("date") >= P.attrs["last_refit"]]
    if oos["fwd_ret"].notna().groupby(level="date").any().sum() < 6:   # too short -> last 36m
        oos = panel[panel.index.get_level_values("date") >= panel.index.get_level_values("date").unique()[-37]]
    ml = list(model_zoo())
    best = ic[ml].mean().idxmax()            # explain the model with the best OOS IC
    imp = bt.permutation_ic_importance(P.attrs["last_models"][best], oos, feats)

    # live signal for the coming month = last date of the panel
    last = P.index.get_level_values("date").max()
    live = P.xs(last, level="date")[strategies].copy()
    w = bt.weights_from_signal(P.loc[[last], "Ensemble"], q=a.quantile, vol=vol).droplevel("date")
    names = {c.yf: c.name for c in UNIVERSE}
    sectors = {c.yf: c.sector for c in UNIVERSE}
    live.insert(0, "weight", w)
    live.insert(0, "sector", live.index.map(sectors))
    live.insert(0, "name", live.index.map(names))
    raw_cols = [c for c in panel.columns if c.startswith("raw_")]
    live = live.join(panel.xs(last, level="date")[raw_cols])

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    P.drop(columns=[], errors="ignore").to_csv(out / "predictions.csv.gz")
    rets.to_csv(out / "returns.csv")
    rets_vt.to_csv(out / "returns_voltarget.csv")
    turns.to_csv(out / "turnover.csv")
    ic.to_csv(out / "ic.csv")
    table.to_csv(out / "summary.csv")
    table_vt.to_csv(out / "summary_voltarget.csv")
    imp.rename("ic_drop").to_csv(out / "importance.csv")
    live.sort_values("Ensemble", ascending=False).to_csv(out / "live_signals.csv")
    panel.to_csv(out / "panel.csv.gz")
    meta = {
        "source": a.source,
        "run_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "signal_date": str(last.date()),
        "first_oos": str(P.index.get_level_values("date").min().date()),
        "n_assets": int(panel.index.get_level_values("ticker").nunique()),
        "features": feats,
        "cot": cot is not None,
        "carry": px2 is not None,
        "refit_every": a.refit_every,
        "quantile": a.quantile,
        "importance_model": best,
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    print(table.round(3).to_string())
    print("\nFeature importance (IC drop):\n", imp.round(4).to_string())


if __name__ == "__main__":
    main()
